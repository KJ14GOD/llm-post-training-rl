import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from lora import build_lora_model
from data import (
    EVAL_GROUND_TRUTH,
    EVAL_LEVEL_COUNTS,
    EVAL_LEVEL_IDS,
    EVAL_PROMPT_QUESTIONS,
    LEVEL_NAMES,
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

    # best case: just one integer
    try:
        return int(body)
    except ValueError:
        pass

    # fallback: split by spaces, go backward, return first int found
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

def smoke_test():
    correct = 0
    total_reward = 0.0
    exact_format = 0
    total_output_len = 0
    level_correct = [0, 0, 0]
    num_questions = len(EVAL_PROMPT_QUESTIONS)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"loading {MODEL_ID} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        attn_implementation="eager"
    )


    model.to(device)
    model.eval()

    for i, (prompt_question, target_answer) in enumerate(zip(EVAL_PROMPT_QUESTIONS, EVAL_GROUND_TRUTH)):
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
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False, # deterministic generation
                pad_token_id=tokenizer.eos_token_id,
            )
        # get input and output lengths
        input_len = inputs["input_ids"].shape[1]
        output_len = output[0].shape[0]

        # print input and output lengths
        print(f"QUESTION: {prompt_question}")
        print(f"input length: {input_len}, output length: {output_len}")

        # get only the newly generated tokens and decode them 
        new_tokens = output[0][input_len:]
        raw_new_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # trim to final response
        trimmed_new_text = trim_to_final_response(raw_new_text)
        final_text = trimmed_new_text if trimmed_new_text != raw_new_text else raw_new_text

        # parse the answer
        parsed_answer = parse_answer(final_text)
        new_token_count = output_len - input_len

        # output length tracking and exact format tracking
        total_output_len += new_token_count
        has_tags = raw_new_text.startswith(START_TAG) and raw_new_text.endswith(END_TAG)
        body = raw_new_text[len(START_TAG):-len(END_TAG)].strip() if has_tags else ""
        is_exact = has_tags and body.lstrip("-").isdigit()

        if is_exact:
            exact_format += 1
        
        
        # reward tracking
        reward_score = reward(parsed_answer, target_answer, trimmed_new_text, new_token_count)
        total_reward += reward_score
        print("NEW TEXT:")
        print(repr(final_text))
        print(f"PARSED ANSWER: {parsed_answer}")
        print(f"GROUND TRUTH: {target_answer}")
        print(f"EXACT FORMAT: {is_exact} | OUTPUT TOKENS: {new_token_count}")
        if (parsed_answer == target_answer):
            print("CORRECT")
            correct+=1
            level_correct[EVAL_LEVEL_IDS[i]] += 1
        else:
            print("INCORRECT")
        print(f"REWARD SCORE: {reward_score}")
        print()
    print("=" * 40)
    for lvl in range(3):
        r = level_correct[lvl]
        level_total = EVAL_LEVEL_COUNTS[lvl]
        print(f"{LEVEL_NAMES[lvl]}: {r}/{level_total} right, {level_total - r}/{level_total} wrong")
    print(f"TOTAL: {correct}/{num_questions} right, {num_questions - correct}/{num_questions} wrong")
    print(f"ACCURACY: {correct / num_questions:.1%}")
    print(f"EXACT FORMAT RATE: {exact_format}/{num_questions} ({exact_format / num_questions:.1%})")
    print(f"AVG OUTPUT LENGTH: {total_output_len / num_questions:.4f} tokens")
    print(f"TOTAL REWARD: {total_reward}")
    print(f"AVG REWARD: {total_reward / num_questions:.4f}")
    
if __name__ == "__main__":
    smoke_test()
