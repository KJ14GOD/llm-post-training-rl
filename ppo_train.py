import random
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from lora import build_lora_model, save_lora_adapters
from data import (
    EVAL_GROUND_TRUTH,
    EVAL_INTERMEDIATES,
    EVAL_LEVEL_COUNTS,
    EVAL_LEVEL_IDS,
    EVAL_PROMPT_QUESTIONS,
    LEVEL_NAMES,
    TRAIN_GROUND_TRUTH,
    TRAIN_INTERMEDIATES,
    TRAIN_LEVEL_COUNTS,
    TRAIN_LEVEL_IDS,
    TRAIN_PROMPT_QUESTIONS,
)
from verifier import END_TAG, START_TAG, is_exact_format, parse_answer, reward, trim_to_final_response

MODEL_ID = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
MAX_NEW_TOKENS = 128
SYSTEM_PROMPT = (
    "You are a precise math assistant. "
    "Solve the problem step by step, showing each operation. "
    "After your reasoning, write the final answer on its own line in this exact format:\n"
    f"{START_TAG}INTEGER{END_TAG}\n"
    "Do not include any text after the closing tag."
)



SEED = 42
NUM_EPOCHS = 50
LEARNING_RATE = 5e-5
CLIP_EPS = 0.2
VALUE_COEF = 0.1
KL_COEF = 0.1
MAX_GRAD_NORM = 1.0
SAMPLE_TEMPERATURE = 0.7
GAE_GAMMA = 0.99
GAE_LAMBDA = 0.95
EVAL_EVERY = 5
PPO_UPDATE_EPOCHS = 4
MINIBATCH_SIZE = 16
ADAPTER_SAVE_PATH = "./lora_adapters"
BEST_ADAPTER_SAVE_PATH = "./lora_adapters_best"


@dataclass
class RolloutRecord:
    prompt_question: str
    target_answer: int
    input_len: int
    output_len: int
    output_ids: torch.Tensor
    generated_token_ids: torch.Tensor
    final_text: str
    parsed_answer: int | None
    output_token_count: int
    exact_format: bool
    policy_logprobs: torch.Tensor
    reference_logprobs: torch.Tensor
    generated_values: torch.Tensor
    final_value_estimate: float
    reward_score: float
    is_correct: bool


@dataclass
class PPOTargets:
    old_policy_logprobs: torch.Tensor
    reference_logprobs: torch.Tensor
    generated_values: torch.Tensor
    reward: float
    returns: torch.Tensor
    advantages: torch.Tensor
    final_advantage: float
    reference_kl_per_token: torch.Tensor
    reference_kl_mean: float


class PolicyWithValueHead(nn.Module):
    def __init__(self, policy_model):
        super().__init__()
        self.policy_model = policy_model
        hidden_size = policy_model.config.hidden_size
        policy_dtype = next(policy_model.parameters()).dtype
        self.value_head = nn.Linear(hidden_size, 1, dtype=policy_dtype)

    def forward(self, *args, **kwargs):
        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True
        outputs = self.policy_model(*args, **kwargs)
        last_hidden_state = outputs.hidden_states[-1]
        values = self.value_head(last_hidden_state).squeeze(-1)
        return outputs, values

    def generate(self, *args, **kwargs):
        return self.policy_model.generate(*args, **kwargs)


def gather_token_logprobs(logits, tokens):
    log_probs = torch.log_softmax(logits, dim=-1) # converting all logits to probabilities 
    token_log_probs = log_probs.gather(dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1) # gather the log probabilities for the tokens 
    return token_log_probs # return the log probabilities for the tokens


def build_ppo_targets(rollout: RolloutRecord):
    old_policy_logprobs = rollout.policy_logprobs
    reference_logprobs = rollout.reference_logprobs
    generated_values = rollout.generated_values # value head's prediction at those positions
    num_tokens = generated_values.shape[0]

    # GAE: walk backward through token positions
    # reward only arrives at the last token, intermediate rewards are 0
    advantages = torch.zeros_like(generated_values)
    gae = 0.0
    for t in reversed(range(num_tokens)):
        if t == num_tokens - 1:
            # last token: delta = reward - V(last)
            next_value = 0.0
            token_reward = rollout.reward_score
        else:
            # earlier tokens: no intermediate reward, just value difference
            next_value = generated_values[t + 1].item()
            token_reward = 0.0
        delta = token_reward + GAE_GAMMA * next_value - generated_values[t].item()
        gae = delta + GAE_GAMMA * GAE_LAMBDA * gae
        advantages[t] = gae

    returns = advantages + generated_values
    final_advantage = rollout.reward_score - rollout.final_value_estimate

    log_ratio = old_policy_logprobs - reference_logprobs
    reference_kl_per_token = (torch.exp(log_ratio) - 1) - log_ratio  # k2 approximate KL estimator
    reference_kl_mean = reference_kl_per_token.mean().item()

    return PPOTargets(
        old_policy_logprobs=old_policy_logprobs,
        reference_logprobs=reference_logprobs,
        generated_values=generated_values,
        reward=rollout.reward_score,
        returns=returns,
        advantages=advantages,
        final_advantage=final_advantage,
        reference_kl_per_token=reference_kl_per_token,
        reference_kl_mean=reference_kl_mean,
    )


def recompute_policy_stats(policy, rollout: RolloutRecord, device):
    output_ids = rollout.output_ids.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(output_ids, device=device)
    policy_outputs, values = policy(input_ids=output_ids, attention_mask=attention_mask)

    policy_logits = policy_outputs.logits[:, :-1, :]
    target_tokens = output_ids[:, 1:]
    policy_token_logprobs = gather_token_logprobs(policy_logits, target_tokens) # pick log prob of the actual next token under the current policy

    generated_policy_logprobs = policy_token_logprobs[:, rollout.input_len - 1:] # generated answer log probs 
    generated_values = values[:, rollout.input_len - 1:-1] # generated answer values
    return generated_policy_logprobs[0], generated_values[0] # return the generated answer log probs and generated value head estimate


def ppo_update(policy, optimizer, rollout_records, ppo_targets, device):
    policy.train()
    num_rollouts = len(rollout_records)
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    policy_losses = []
    value_losses = []
    kl_terms = []
    total_losses = []

    for _ in range(PPO_UPDATE_EPOCHS):
        shuffled_indices = list(range(num_rollouts))
        random.shuffle(shuffled_indices)

        for start_idx in range(0, num_rollouts, MINIBATCH_SIZE):
            minibatch_indices = shuffled_indices[start_idx:start_idx + MINIBATCH_SIZE]
            minibatch_policy_losses = []
            minibatch_value_losses = []
            minibatch_kl_terms = []
            minibatch_total_losses = []

            for idx in minibatch_indices:
                rollout = rollout_records[idx]
                targets = ppo_targets[idx]

                new_policy_logprobs, new_values = recompute_policy_stats(policy, rollout, device)
                old_policy_logprobs = targets.old_policy_logprobs.to(device)
                reference_logprobs = targets.reference_logprobs.to(device)
                advantages = targets.advantages.to(device)
                returns = targets.returns.to(device)

                ratios = torch.exp(new_policy_logprobs - old_policy_logprobs)
                clipped_ratios = torch.clamp(ratios, 1 - CLIP_EPS, 1 + CLIP_EPS)

                unclipped_objective = ratios * advantages
                clipped_objective = clipped_ratios * advantages
                policy_loss = -torch.min(unclipped_objective, clipped_objective).mean()
                value_loss = F.mse_loss(new_values, returns)
                kl_log_ratio = new_policy_logprobs - reference_logprobs
                reference_kl = ((torch.exp(kl_log_ratio) - 1) - kl_log_ratio).mean()
                total_loss = policy_loss + (VALUE_COEF * value_loss) + (KL_COEF * reference_kl)

                minibatch_policy_losses.append(policy_loss)
                minibatch_value_losses.append(value_loss)
                minibatch_kl_terms.append(reference_kl)
                minibatch_total_losses.append(total_loss)

            mean_policy_loss = torch.stack(minibatch_policy_losses).mean()
            mean_value_loss = torch.stack(minibatch_value_losses).mean()
            mean_reference_kl = torch.stack(minibatch_kl_terms).mean()
            mean_total_loss = torch.stack(minibatch_total_losses).mean()

            optimizer.zero_grad()
            mean_total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
            optimizer.step()

            policy_losses.append(mean_policy_loss.detach())
            value_losses.append(mean_value_loss.detach())
            kl_terms.append(mean_reference_kl.detach())
            total_losses.append(mean_total_loss.detach())

    policy.eval()

    return {
        "policy_loss": torch.stack(policy_losses).mean().item(),
        "value_loss": torch.stack(value_losses).mean().item(),
        "reference_kl": torch.stack(kl_terms).mean().item(),
        "total_loss": torch.stack(total_losses).mean().item(),
        "ppo_update_epochs": PPO_UPDATE_EPOCHS,
        "minibatch_size": MINIBATCH_SIZE,
    }


def collect_rollout_record(policy, reference_model, tokenizer, prompt_question, target_answer, device, do_sample=True, intermediate_results=None):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    generate_kwargs = dict(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = SAMPLE_TEMPERATURE
    else:
        generate_kwargs["do_sample"] = False

    with torch.no_grad():
        output = policy.generate(**generate_kwargs)

    input_len = inputs["input_ids"].shape[1]
    output_len = output[0].shape[0]
    new_tokens = output[0][input_len:]
    raw_new_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    trimmed_new_text = trim_to_final_response(raw_new_text)
    parsed_answer = parse_answer(raw_new_text)
    output_token_count = output_len - input_len

    full_attention_mask = torch.ones_like(output, device=device) # attentin mask 
    with torch.no_grad(): # inference no gradients
        policy_outputs, values = policy(input_ids=output, attention_mask=full_attention_mask)
        reference_outputs = reference_model(
            input_ids=output,
            attention_mask=full_attention_mask,
            return_dict=True,
        )

    policy_logits = policy_outputs.logits[:, :-1, :] # logits for the policy model [batch_size, sequence_length, vocab_size], drop the last token because there is no next token to predict
    reference_logits = reference_outputs.logits[:, :-1, :] 
    target_tokens = output[:, 1:] # drop the first token because there is no previous token to predict

    # output = [A,B,C,D]
    # logits row 0 logits row 0 = prediction for B after seeing A
    # logits row 1 = prediction for C after seeing A, B
    # logits row 2 = prediction for D after seeing A, B, C
    # logits row 3 = prediction for the token after D
    # after manipulation perfect matching
    # logits row 0 compares to B
    # logits row 1 compares to C
    # logits row 2 compares to D

    policy_token_logprobs = gather_token_logprobs(policy_logits, target_tokens)
    reference_token_logprobs = gather_token_logprobs(reference_logits, target_tokens)

    generated_policy_logprobs = policy_token_logprobs[:, input_len - 1:]
    generated_reference_logprobs = reference_token_logprobs[:, input_len - 1:]
    generated_values = values[:, input_len - 1:-1]
    final_value_estimate = values[0, -1].item()

    is_exact = is_exact_format(raw_new_text)

    reward_score = reward(parsed_answer, target_answer, raw_new_text, output_token_count, intermediate_results=intermediate_results)
    is_correct = parsed_answer == target_answer

    return RolloutRecord(
        prompt_question=prompt_question,
        target_answer=target_answer,
        input_len=input_len,
        output_len=output_len,
        output_ids=output[0].detach().cpu(),
        generated_token_ids=new_tokens.detach().cpu(),
        final_text=trimmed_new_text,
        parsed_answer=parsed_answer,
        output_token_count=output_token_count,
        exact_format=is_exact,
        policy_logprobs=generated_policy_logprobs[0].detach().cpu(),
        reference_logprobs=generated_reference_logprobs[0].detach().cpu(),
        generated_values=generated_values[0].detach().cpu(),
        final_value_estimate=final_value_estimate,
        reward_score=reward_score,
        is_correct=is_correct,
    )


def evaluate_policy(policy, tokenizer, device, label="EVAL"):
    """Run deterministic evaluation on all questions. Returns metrics dict."""
    policy.eval()
    correct = 0
    exact_format_count = 0
    total_reward = 0.0
    total_output_len = 0
    level_correct = [0, 0, 0]
    num_questions = len(EVAL_PROMPT_QUESTIONS)

    for i, (prompt_question, target_answer, intermediates) in enumerate(zip(EVAL_PROMPT_QUESTIONS, EVAL_GROUND_TRUTH, EVAL_INTERMEDIATES)):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_question},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = policy.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        output_len = output[0].shape[0]
        new_tokens = output[0][input_len:]
        raw_new_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        trimmed_new_text = trim_to_final_response(raw_new_text)
        parsed_answer = parse_answer(raw_new_text)
        new_token_count = output_len - input_len

        total_output_len += new_token_count
        is_exact = is_exact_format(raw_new_text)

        if is_exact:
            exact_format_count += 1

        reward_score = reward(parsed_answer, target_answer, raw_new_text, new_token_count, intermediate_results=intermediates)
        total_reward += reward_score

        is_correct = parsed_answer == target_answer
        if is_correct:
            correct += 1
            level_correct[EVAL_LEVEL_IDS[i]] += 1

        status = "CORRECT" if is_correct else "INCORRECT"
        print(f"  [{status}] Q: {prompt_question[:60]}  A: {repr(trimmed_new_text)[:40]}  parsed={parsed_answer}  truth={target_answer}  reward={reward_score}")

    print(f"\n{'=' * 40}")
    print(f"  {label}")
    print(f"{'=' * 40}")
    for lvl in range(3):
        r = level_correct[lvl]
        print(f"  {LEVEL_NAMES[lvl]}: {r}/{EVAL_LEVEL_COUNTS[lvl]}")
    print(f"  ACCURACY: {correct}/{num_questions} ({correct / num_questions:.1%})")
    print(f"  EXACT FORMAT RATE: {exact_format_count}/{num_questions} ({exact_format_count / num_questions:.1%})")
    print(f"  AVG OUTPUT LENGTH: {total_output_len / num_questions:.1f} tokens")
    print(f"  AVG REWARD: {total_reward / num_questions:.4f}")
    print(f"{'=' * 40}\n")

    return {
        "correct": correct,
        "accuracy": correct / num_questions,
        "exact_format": exact_format_count,
        "exact_format_rate": exact_format_count / num_questions,
        "avg_output_len": total_output_len / num_questions,
        "avg_reward": total_reward / num_questions,
        "level_correct": level_correct,
    }


def train_ppo_epoch(policy, reference_model, tokenizer, optimizer, device, epoch):
    """Collect rollouts with sampling and run one PPO update. Returns stats."""
    rollout_records = []
    ppo_targets_list = []
    num_correct = 0
    total_reward = 0.0

    for i, (prompt_question, target_answer, intermediates) in enumerate(zip(TRAIN_PROMPT_QUESTIONS, TRAIN_GROUND_TRUTH, TRAIN_INTERMEDIATES)):
        rollout = collect_rollout_record(
            policy=policy,
            reference_model=reference_model,
            tokenizer=tokenizer,
            prompt_question=prompt_question,
            target_answer=target_answer,
            device=device,
            do_sample=True,
            intermediate_results=intermediates,
        )
        targets = build_ppo_targets(rollout)
        rollout_records.append(rollout)
        ppo_targets_list.append(targets)

        if rollout.is_correct:
            num_correct += 1
        total_reward += rollout.reward_score

        if not rollout.is_correct:
            print(f"    [INCORRECT] Q: {prompt_question}")
            print(f"      Full output: {repr(rollout.final_text)}")
            print(f"      Parsed: {rollout.parsed_answer}  Truth: {target_answer}  Reward: {rollout.reward_score}")

    num_questions = len(TRAIN_PROMPT_QUESTIONS)
    print(f"  Epoch {epoch + 1} rollouts: {num_correct}/{num_questions} correct, avg reward {total_reward / num_questions:.4f}")

    # normalize advantages across the batch
    all_advantages = torch.cat([t.advantages for t in ppo_targets_list])
    adv_mean = all_advantages.mean()
    adv_std = all_advantages.std()
    for t in ppo_targets_list:
        t.advantages = (t.advantages - adv_mean) / (adv_std + 1e-8)

    update_stats = ppo_update(policy, optimizer, rollout_records, ppo_targets_list, device)
    print(
        f"  Epoch {epoch + 1} update: "
        f"policy_loss={update_stats['policy_loss']:.4f} "
        f"value_loss={update_stats['value_loss']:.4f} "
        f"kl={update_stats['reference_kl']:.4f} "
        f"total_loss={update_stats['total_loss']:.4f} "
        f"(ppo_passes={update_stats['ppo_update_epochs']}, minibatch_size={update_stats['minibatch_size']})"
    )

    return update_stats


def main():
    # set seeds for reproducibility
    random.seed(SEED)
    torch.manual_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # build policy with LoRA + value head
    lora_model = build_lora_model(MODEL_ID, dtype)
    policy = PolicyWithValueHead(lora_model).to(device) 
    policy.eval()

    optimizer = torch.optim.Adam(
        [param for param in policy.parameters() if param.requires_grad],
        lr=LEARNING_RATE,
    )

    # frozen reference model for KL penalty
    reference_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(device)

    for param in reference_model.parameters():
        param.requires_grad = False
    reference_model.eval()

    # pre-training evaluation
    print("Pre-training evaluation...")
    pre_metrics = evaluate_policy(policy, tokenizer, device, label="PRE-TRAINING BASELINE")
    best_reward = pre_metrics["avg_reward"]

    # training loop
    print(f"Starting PPO training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        train_ppo_epoch(policy, reference_model, tokenizer, optimizer, device, epoch)

        if (epoch + 1) % EVAL_EVERY == 0:
            metrics = evaluate_policy(policy, tokenizer, device, label=f"EVAL EPOCH {epoch + 1}")
            if metrics["avg_reward"] > best_reward:
                best_reward = metrics["avg_reward"]
                save_lora_adapters(policy.policy_model, BEST_ADAPTER_SAVE_PATH)
                print(f"  New best reward: {best_reward:.4f} — saved to {BEST_ADAPTER_SAVE_PATH}")

    # save latest adapters
    save_lora_adapters(policy.policy_model, ADAPTER_SAVE_PATH)

    # post-training evaluation (latest checkpoint)
    print("Post-training evaluation (LATEST)...")
    latest_metrics = evaluate_policy(policy, tokenizer, device, label="POST-TRAINING (LATEST)")

    # post-training evaluation (best checkpoint)
    best_metrics = None
    if os.path.exists(BEST_ADAPTER_SAVE_PATH):
        print("Loading best checkpoint for evaluation...")
        from lora import load_lora_adapters
        best_lora_model = load_lora_adapters(MODEL_ID, BEST_ADAPTER_SAVE_PATH, dtype)
        best_policy = PolicyWithValueHead(best_lora_model).to(device)
        best_policy.eval()
        best_metrics = evaluate_policy(best_policy, tokenizer, device, label="POST-TRAINING (BEST)")
        del best_policy, best_lora_model

    # comparison
    print("=" * 50)
    print("  BASELINE vs LATEST vs BEST")
    print("=" * 50)
    header = f"  {'':20s} {'Baseline':>10s}  {'Latest':>10s}"
    acc_line = f"  {'Accuracy':20s} {pre_metrics['accuracy']:>10.1%}  {latest_metrics['accuracy']:>10.1%}"
    fmt_line = f"  {'Exact format':20s} {pre_metrics['exact_format_rate']:>10.1%}  {latest_metrics['exact_format_rate']:>10.1%}"
    rew_line = f"  {'Avg reward':20s} {pre_metrics['avg_reward']:>10.4f}  {latest_metrics['avg_reward']:>10.4f}"
    len_line = f"  {'Avg length':20s} {pre_metrics['avg_output_len']:>10.1f}  {latest_metrics['avg_output_len']:>10.1f}"
    if best_metrics:
        header += f"  {'Best':>10s}"
        acc_line += f"  {best_metrics['accuracy']:>10.1%}"
        fmt_line += f"  {best_metrics['exact_format_rate']:>10.1%}"
        rew_line += f"  {best_metrics['avg_reward']:>10.4f}"
        len_line += f"  {best_metrics['avg_output_len']:>10.1f}"
    print(header)
    print(f"  {'─' * (len(header) - 2)}")
    print(acc_line)
    print(fmt_line)
    print(rew_line)
    print(len_line)
    print("=" * 50)


if __name__ == "__main__":
    main()
