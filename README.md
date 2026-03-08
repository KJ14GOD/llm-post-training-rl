# RL Training Scratchpad

This repo is currently in the baseline stage for a small RL post-training project.

Right now the goal is not PPO yet. The goal is to:

1. Load a pretrained instruct model locally.
2. Verify it runs on the laptop.
3. Probe its behavior on a fixed set of math questions.
4. Measure what it gets right, wrong, or formats badly.

Once that baseline is clear, the next step is to add a verifier and compare base model vs PPO-trained LoRA policy.

## Current model

- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- Current script: [train.py](/Users/kj16/Desktop/rl_training/train.py)

## What `train.py` does right now

The script currently:

- loads the pretrained Qwen model
- loads the tokenizer
- runs through a list of math prompts
- generates a response for each prompt
- prints prompt/output length information
- slices off the prompt tokens and prints only the newly generated text

Important: this is currently **sequential inference**, not batching.

That means the code does:

- one prompt
- one model call
- one output
- repeat

Batch size is effectively `1` on each loop iteration.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install torch transformers accelerate sentencepiece peft huggingface_hub
```

Optional:

- set `HF_TOKEN` if you want faster Hugging Face downloads and higher rate limits

## Run

```bash
python train.py
```

## Interpreting the tensors

For a single prompt:

- `inputs["input_ids"].shape` is usually `[1, sequence_length]`
- `1` is the batch size
- `sequence_length` is the number of prompt tokens

For generated output:

- `output.shape` is usually `[1, total_sequence_length]`
- `output[0]` is the full first sequence
- that first sequence includes both:
  - the original prompt tokens
  - the newly generated tokens

To get only the generated continuation:

```python
input_len = inputs["input_ids"].shape[1]
new_tokens = output[0][input_len:]
new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
```

## Why outputs look weird

A pretrained instruct model is not guaranteed to answer with just `45` even if the prompt says "Answer with only the integer."

It may generate things like:

- extra spaces
- a sentence instead of a bare number
- a worked solution
- the wrong answer

That is normal. This is exactly why the baseline stage matters.

The project is not just about raw correctness. It is also about:

- format control
- concise answering
- verifier-friendly outputs

## Current known issues

### MPS on the M2 Air

Running the model on `mps` hit an Apple MPS allocation error:

- `Error: total bytes of NDArray > 2**32`

The same script worked on `cpu`, so the current practical baseline path is CPU inference.

### Generation warnings

You may see:

```text
The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']
```

That comes from the model's default generation config and usually happens when `do_sample=False`.
It is not the main problem.

### Long rambling outputs

If `max_new_tokens` is too high, the model may ramble instead of answering cleanly.

For strict math-answer tasks, smaller values like `8`, `16`, or `32` are usually better than `512`.

## Immediate next steps

Before adding PPO:

1. Reduce `max_new_tokens` to something small.
2. Parse the model's generated text into an integer answer.
3. Compare that parsed answer to the ground truth.
4. Record simple metrics:
   - correctness
   - exact-format rate
   - output length

After that baseline is working, the next stage is:

1. define a verifier reward
2. add LoRA adapters
3. run on-policy PPO rollouts
4. compare base vs PPO-trained model

## Notes

- `repr(text)` is useful for debugging model outputs because it shows hidden characters like `\n`
- `output[0]` is not "the answer only"; it is the full first generated sequence
- true batching would pass a list of prompts to the tokenizer at once instead of looping prompt-by-prompt
