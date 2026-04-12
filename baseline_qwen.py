import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from data import (
    EVAL_GROUND_TRUTH,
    EVAL_LEVEL_COUNTS,
    EVAL_LEVEL_IDS,
    EVAL_PROMPT_QUESTIONS,
    LEVEL_NAMES,
)
from verifier import END_TAG, START_TAG, is_exact_format, parse_answer, reward, trim_to_final_response

MODEL_ID = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
MAX_NEW_TOKENS = 64
SYSTEM_PROMPT = (
    "You are a precise math assistant. "
    "Solve the problem step by step, showing each operation. "
    "After your reasoning, write the final answer on its own line in this exact format:\n"
    f"{START_TAG}INTEGER{END_TAG}\n"
    "Do not include any text after the closing tag."
)

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
                max_new_tokens=MAX_NEW_TOKENS,
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
        trimmed_new_text = trim_to_final_response(raw_new_text)

        parsed_answer = parse_answer(raw_new_text)
        new_token_count = output_len - input_len

        total_output_len += new_token_count
        is_exact = is_exact_format(raw_new_text)

        if is_exact:
            exact_format += 1

        reward_score = reward(parsed_answer, target_answer, raw_new_text, new_token_count)
        total_reward += reward_score
        print("NEW TEXT:")
        print(repr(trimmed_new_text))
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
