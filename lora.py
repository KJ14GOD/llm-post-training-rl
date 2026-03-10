from peft import LoraConfig, TaskType, get_peft_model, PeftModel
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


def save_lora_adapters(model, path):
    model.save_pretrained(path)
    print(f"LoRA adapters saved to {path}")


def load_lora_adapters(model_id, adapter_path, dtype):
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    print(f"LoRA adapters loaded from {adapter_path}")
    return model




