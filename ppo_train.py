import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from lora import build_lora_model, save_lora_adapters
from data import (
    EVAL_GROUND_TRUTH,
    EVAL_LEVEL_COUNTS,
    EVAL_LEVEL_IDS,
    EVAL_PROMPT_QUESTIONS,
    LEVEL_NAMES,
    TRAIN_GROUND_TRUTH,
    TRAIN_LEVEL_COUNTS,
    TRAIN_LEVEL_IDS,
    TRAIN_PROMPT_QUESTIONS,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
START_TAG = "<final_response>"
END_TAG = "</final_response>"
SYSTEM_PROMPT = (
    "You are a precise math assistant. "
    "Return exactly one XML block in this format: "
    "<final_response>INTEGER</final_response>. "
    "Do not show reasoning or examples."
)



SEED = 42
NUM_EPOCHS = 50
LEARNING_RATE = 5e-5
CLIP_EPS = 0.2
VALUE_COEF = 0.1
KL_COEF = 0.01
MAX_GRAD_NORM = 1.0
SAMPLE_TEMPERATURE = 0.7
ADAPTER_SAVE_PATH = "./lora_adapters"


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


def reward(parsed_answer, ground_truth, raw_text, num_tokens):
    stripped = raw_text.strip()
    bare_integer=stripped.lstrip("-").isdigit()
    score = 0.0
    has_tags = raw_text.startswith(START_TAG) and raw_text.endswith(END_TAG)
    body = raw_text[len(START_TAG):-len(END_TAG)].strip() if has_tags else ""
    tagged_integer = has_tags and body.lstrip("-").isdigit()
    exact_format = bare_integer or tagged_integer

    if parsed_answer is None:
        score -= 0.5
    elif parsed_answer == ground_truth and exact_format:
        score += 1.0
    elif parsed_answer == ground_truth:
        score += 0.5
    else:
        score -= 0.25

    if parsed_answer == ground_truth and exact_format and num_tokens <= 10:
        score += 0.05

    return score


def parse_answer(text):
    if text.startswith(START_TAG) and text.endswith(END_TAG):
        body = text[len(START_TAG):-len(END_TAG)].strip()
    else:
        body = text.strip()

    try:
        return int(body)
    except ValueError:
        pass

    parts = body.split()
    for part in reversed(parts):
        cleaned = part.strip(".,!?()[]{}")
        try:
            return int(cleaned)
        except ValueError:
            continue

    return None


def trim_to_final_response(text):
    if END_TAG not in text:
        return text

    return text.split(END_TAG, 1)[0] + END_TAG


def build_ppo_targets(rollout: RolloutRecord):
    old_policy_logprobs = rollout.policy_logprobs 
    reference_logprobs = rollout.reference_logprobs
    generated_values = rollout.generated_values # value head's prediction at those positions

    reward_tensor = torch.tensor(rollout.reward_score, dtype=generated_values.dtype)
    returns = torch.full_like(generated_values, fill_value=reward_tensor.item())  # input shape of generated values and filled with reward_tensor value
    advantages = returns - generated_values # actual - predicted
    final_advantage = rollout.reward_score - rollout.final_value_estimate

    reference_kl_per_token = old_policy_logprobs - reference_logprobs # KL penalty for each token
    reference_kl_mean = reference_kl_per_token.mean().item() # average KL penalty

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
    policy_losses = []
    value_losses = []
    kl_terms = []
    total_losses = []

    for rollout, targets in zip(rollout_records, ppo_targets):
        new_policy_logprobs, new_values = recompute_policy_stats(policy, rollout, device)
        old_policy_logprobs = targets.old_policy_logprobs.to(device)
        reference_logprobs = targets.reference_logprobs.to(device)
        advantages = targets.advantages.to(device)
        returns = targets.returns.to(device)

        ratios = torch.exp(new_policy_logprobs - old_policy_logprobs) # ratio of the new policy log probs over the old policy log probs
        clipped_ratios = torch.clamp(ratios, 1 - CLIP_EPS, 1 + CLIP_EPS) # bound the ratio between 0.8 and 1.2

        unclipped_objective = ratios * advantages # without clipping
        clipped_objective = clipped_ratios * advantages
        policy_loss = -torch.min(unclipped_objective, clipped_objective).mean()
        value_loss = F.mse_loss(new_values, returns)
        reference_kl = (new_policy_logprobs - reference_logprobs).mean()

        total_loss = policy_loss + (VALUE_COEF * value_loss) + (KL_COEF * reference_kl)

        policy_losses.append(policy_loss)
        value_losses.append(value_loss)
        kl_terms.append(reference_kl)
        total_losses.append(total_loss)

    mean_policy_loss = torch.stack(policy_losses).mean()
    mean_value_loss = torch.stack(value_losses).mean()
    mean_reference_kl = torch.stack(kl_terms).mean()
    mean_total_loss = torch.stack(total_losses).mean()

    optimizer.zero_grad()
    mean_total_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in policy.parameters() if p.requires_grad],
        MAX_GRAD_NORM,
    )
    optimizer.step()
    policy.eval()

    return {
        "policy_loss": mean_policy_loss.item(),
        "value_loss": mean_value_loss.item(),
        "reference_kl": mean_reference_kl.item(),
        "total_loss": mean_total_loss.item(),
    }


def collect_rollout_record(policy, reference_model, tokenizer, prompt_question, target_answer, device, do_sample=True):
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
        max_new_tokens=32,
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
    raw_new_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip() # convert to string 
    trimmed_new_text = trim_to_final_response(raw_new_text) # trim to final response
    final_text = trimmed_new_text if trimmed_new_text != raw_new_text else raw_new_text
    parsed_answer = parse_answer(final_text)
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

    has_tags = raw_new_text.startswith(START_TAG) and raw_new_text.endswith(END_TAG)
    body = raw_new_text[len(START_TAG):-len(END_TAG)].strip() if has_tags else ""
    is_exact = has_tags and body.lstrip("-").isdigit()

    reward_score = reward(parsed_answer, target_answer, trimmed_new_text, output_token_count)
    is_correct = parsed_answer == target_answer

    return RolloutRecord(
        prompt_question=prompt_question,
        target_answer=target_answer,
        input_len=input_len,
        output_len=output_len,
        output_ids=output[0].detach().cpu(),
        generated_token_ids=new_tokens.detach().cpu(),
        final_text=final_text,
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

    for i, (prompt_question, target_answer) in enumerate(zip(EVAL_PROMPT_QUESTIONS, EVAL_GROUND_TRUTH)):
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
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        output_len = output[0].shape[0]
        new_tokens = output[0][input_len:]
        raw_new_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        trimmed_new_text = trim_to_final_response(raw_new_text)
        final_text = trimmed_new_text if trimmed_new_text != raw_new_text else raw_new_text
        parsed_answer = parse_answer(final_text)
        new_token_count = output_len - input_len

        total_output_len += new_token_count
        has_tags = raw_new_text.startswith(START_TAG) and raw_new_text.endswith(END_TAG)
        body = raw_new_text[len(START_TAG):-len(END_TAG)].strip() if has_tags else ""
        is_exact = has_tags and body.lstrip("-").isdigit()

        if is_exact:
            exact_format_count += 1

        reward_score = reward(parsed_answer, target_answer, trimmed_new_text, new_token_count)
        total_reward += reward_score

        if parsed_answer == target_answer:
            correct += 1
            level_correct[EVAL_LEVEL_IDS[i]] += 1

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

    for i, (prompt_question, target_answer) in enumerate(zip(TRAIN_PROMPT_QUESTIONS, TRAIN_GROUND_TRUTH)):
        rollout = collect_rollout_record(
            policy=policy,
            reference_model=reference_model,
            tokenizer=tokenizer,
            prompt_question=prompt_question,
            target_answer=target_answer,
            device=device,
            do_sample=True,
        )
        targets = build_ppo_targets(rollout)
        rollout_records.append(rollout)
        ppo_targets_list.append(targets)

        if rollout.is_correct:
            num_correct += 1
        total_reward += rollout.reward_score

    num_questions = len(TRAIN_PROMPT_QUESTIONS)
    print(f"  Epoch {epoch + 1} rollouts: {num_correct}/{num_questions} correct, avg reward {total_reward / num_questions:.4f}")

    update_stats = ppo_update(policy, optimizer, rollout_records, ppo_targets_list, device)
    print(f"  Epoch {epoch + 1} update: policy_loss={update_stats['policy_loss']:.4f} value_loss={update_stats['value_loss']:.4f} kl={update_stats['reference_kl']:.4f} total_loss={update_stats['total_loss']:.4f}")

    return update_stats


def main():
    # set seeds for reproducibility
    random.seed(SEED)
    torch.manual_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

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

    # training loop
    print(f"Starting PPO training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        train_ppo_epoch(policy, reference_model, tokenizer, optimizer, device, epoch)

    # post-training evaluation
    print("Post-training evaluation...")
    post_metrics = evaluate_policy(policy, tokenizer, device, label="POST-TRAINING")

    # comparison
    print("=" * 40)
    print("  BEFORE vs AFTER")
    print("=" * 40)
    print(f"  Accuracy:     {pre_metrics['accuracy']:.1%} -> {post_metrics['accuracy']:.1%}")
    print(f"  Exact format: {pre_metrics['exact_format_rate']:.1%} -> {post_metrics['exact_format_rate']:.1%}")
    print(f"  Avg reward:   {pre_metrics['avg_reward']:.4f} -> {post_metrics['avg_reward']:.4f}")
    print(f"  Avg length:   {pre_metrics['avg_output_len']:.1f} -> {post_metrics['avg_output_len']:.1f}")
    print("=" * 40)

    # save trained adapters
    save_lora_adapters(policy.policy_model, ADAPTER_SAVE_PATH)


if __name__ == "__main__":
    main()
