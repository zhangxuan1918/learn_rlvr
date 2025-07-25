from unsloth import FastLanguageModel
from peft import PeftModel

def get_model(model_name: str, max_seq_length: int, load_in_4bit: bool, fast_inference: bool, lora_rank: int, gpu_memory_utilization: float):
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=fast_inference,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=gpu_memory_utilization
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_rank * 2,
        use_gradient_checkpointing=True,
        random_state=42,
        )
    return model, tokenizer


def load_lora_adapter(
    model_base, lora_adapter_path: str, adapter_name: str = "default", is_trainable: bool = False
):
    return PeftModel.from_pretrained(
        model=model_base,
        model_id=lora_adapter_path,
        adapter_name=adapter_name,
        is_trainable=is_trainable
    )