from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM
import torch


def build_lora_model(model_id, dtype):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r = 8,
        lora_alpha = 16,
        lora_dropout = 0.05,
        target_modules = ["q_proj", "v_proj"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        attn_implementation="eager",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model




