import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
START_TAG = "<final_response>"
END_TAG = "</final_response>"
SYSTEM_PROMPT = (
    "You are a precise math assistant. "
    "Return exactly one XML block in this format: "
    "<final_response>INTEGER</final_response>. "
    "Do not show reasoning or examples."
)


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


def get_level_name(index):
    if index < 10:
        return "Level 1: Basic Arithmetic"
    if index < 20:
        return "Level 2: Multi-Step Arithmetic"
    return "Level 3: Word Problems"

def smoke_test():
    # device = "mps" if torch.backends.mps.is_available() else "cpu" 
    correct = 0
    answers = [
        45, 23, 504, 12, 86, 35, 391, 14, 83, 33,
        180, 138, 624, 108, 53, 245, 474, 90, 199, 280,
        180, 192, 41, 1728, 78, 84, 288, 432, 1620, 87,
    ]
    level_stats = {
        "Level 1: Basic Arithmetic": {"correct": 0, "total": 0},
        "Level 2: Multi-Step Arithmetic": {"correct": 0, "total": 0},
        "Level 3: Word Problems": {"correct": 0, "total": 0},
    }
    device = "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32

    print(f"loading {MODEL_ID} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        attn_implementation="eager"
    )
    model.to(device)
    model.eval()

    prompt_questions = [
        # Level 1: Basic Arithmetic
        "What is 17 + 28?",
        "What is 42 - 19?",
        "What is 36 * 14?",
        "What is 144 / 12?",
        "What is 29 + 57?",
        "What is 81 - 46?",
        "What is 23 * 17?",
        "What is 196 / 14?",
        "What is 45 + 38?",
        "What is 72 - 39?",
        # Level 2: Multi-Step Arithmetic
        "What is (17 + 28) * 4?",
        "What is (42 - 19) * 6?",
        "What is (36 * 14) + 120?",
        "What is (144 / 12) * 9?",
        "What is (29 + 57) - 33?",
        "What is (81 - 46) * 7?",
        "What is (23 * 17) + (45 + 38)?",
        "What is (196 / 14) * 5 + 20?",
        "What is (45 + 38) * 3 - 50?",
        "What is (72 - 39) * 8 + 16?",
        # Level 3: Word Problems
        "A train travels 60 miles per hour for 3 hours. How far does it travel?",
        "A box contains 24 packs of pencils and each pack has 8 pencils. How many pencils are there in total?",
        "Sarah buys 5 notebooks for 7 dollars each and 3 pens for 2 dollars each. What is the total cost in dollars?",
        "A warehouse stores 48 boxes and each box contains 36 items. How many items are stored in total?",
        "A farmer has 125 apples and sells 47 of them. How many apples remain?",
        "A classroom has 28 students and each student receives 3 books. How many books are distributed in total?",
        "A car travels 72 miles per hour for 4 hours. How far does it travel?",
        "A store sells 18 boxes of cookies and each box contains 24 cookies. How many cookies were sold?",
        "A school buys 36 desks and each desk costs 45 dollars. What is the total cost in dollars?",
        "A delivery truck carries 125 packages and delivers 38 of them. How many packages remain?"
    ]

    assert len(prompt_questions) == len(answers)

    for idx, (prompt_question, target_answer) in enumerate(zip(prompt_questions, answers)):
        level_name = get_level_name(idx)
        level_stats[level_name]["total"] += 1
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
                max_new_tokens=512,
                do_sample=False, # deterministic generation
                pad_token_id=tokenizer.eos_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        output_len = output[0].shape[0]
        print(f"QUESTION: {prompt_question}")
        print(f"LEVEL: {level_name}")
        print(f"input length: {input_len}, output length: {output_len}")
        new_tokens = output[0][input_len:]
        raw_new_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        trimmed_new_text = trim_to_final_response(raw_new_text)
        final_text = trimmed_new_text if trimmed_new_text != raw_new_text else raw_new_text
        parsed_answer = parse_answer(final_text)
        print("NEW TEXT:")
        print(repr(final_text))
        print(f"PARSED ANSWER: {parsed_answer}")
        print(f"GROUND TRUTH: {target_answer}")
        if parsed_answer == target_answer:
            print("CORRECT")
            correct += 1
            level_stats[level_name]["correct"] += 1
        else:
            print("INCORRECT")
        print()
    print(f"CORRECT: {correct}, INCORRECT: {len(prompt_questions) - correct}")
    print(f"ACCURACY: {correct / len(prompt_questions)}")
    print()
    for level_name, stats in level_stats.items():
        level_accuracy = stats["correct"] / stats["total"]
        print(f"{level_name}: {stats['correct']}/{stats['total']} = {level_accuracy:.1%}")

    
if __name__ == "__main__":
    smoke_test()
