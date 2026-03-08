import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

def smoke_test():
    # device = "mps" if torch.backends.mps.is_available() else "cpu" 
    device = "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32

    print(f"loading {MODEL_ID} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        attn_implementation="eager"
    )
    model.to("cpu")
    model.eval()

    prompt_questions = [
        # Level 1: Basic Arithmetic
        "What is 17 + 28? Answer with only the integer.",
        "What is 42 - 19? Answer with only the integer.",
        "What is 36 * 14? Answer with only the integer.",
        "What is 144 / 12? Answer with only the integer.",
        "What is 29 + 57? Answer with only the integer.",
        "What is 81 - 46? Answer with only the integer.",
        "What is 23 * 17? Answer with only the integer.",
        "What is 196 / 14? Answer with only the integer.",
        "What is 45 + 38? Answer with only the integer.",
        "What is 72 - 39? Answer with only the integer.",
        # Level 2: Multi-Step Arithmetic
        "What is (17 + 28) * 4? Answer with only the integer.",
        "What is (42 - 19) * 6? Answer with only the integer.",
        "What is (36 * 14) + 120? Answer with only the integer.",
        "What is (144 / 12) * 9? Answer with only the integer.",
        "What is (29 + 57) - 33? Answer with only the integer.",
        "What is (81 - 46) * 7? Answer with only the integer.",
        "What is (23 * 17) + (45 + 38)? Answer with only the integer.",
        "What is (196 / 14) * 5 + 20? Answer with only the integer.",
        "What is (45 + 38) * 3 - 50? Answer with only the integer.",
        "What is (72 - 39) * 8 + 16? Answer with only the integer.",
        # Level 3: Word Problems
        "A train travels 60 miles per hour for 3 hours. How far does it travel? Answer with only the integer.",
        "A box contains 24 packs of pencils and each pack has 8 pencils. How many pencils are there in total? Answer with only the integer.",
        "Sarah buys 5 notebooks for 7 dollars each and 3 pens for 2 dollars each. What is the total cost in dollars? Answer with only the integer.",
        "A warehouse stores 48 boxes and each box contains 36 items. How many items are stored in total? Answer with only the integer.",
        "A farmer has 125 apples and sells 47 of them. How many apples remain? Answer with only the integer.",
        "A classroom has 28 students and each student receives 3 books. How many books are distributed in total? Answer with only the integer.",
        "A car travels 72 miles per hour for 4 hours. How far does it travel? Answer with only the integer.",
        "A store sells 18 boxes of cookies and each box contains 24 cookies. How many cookies were sold? Answer with only the integer.",
        "A school buys 36 desks and each desk costs 45 dollars. What is the total cost in dollars? Answer with only the integer.",
        "A delivery truck carries 125 packages and delivers 38 of them. How many packages remain? Answer with only the integer."
    ]

    # inputs = tokenizer(prompt_questions, return_tensors="pt").to(device)
   

    for prompt_question in prompt_questions:
        inputs = tokenizer(prompt_question, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False, # deterministic generation
            )
        input_len = inputs["input_ids"].shape[1]
        output_len = output[0].shape[0]
        print(f"input length: {input_len}, output length: {output_len}")
        new_tokens = output[0][input_len:]
        new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        print("NEW TEXT:")
        print(repr(new_text))

    
if __name__ == "__main__":
    smoke_test()
