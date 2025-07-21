
import torch
from data import SYSTEM_PROMPT_DETAILED, SYSTEM_PROMPT
from inference_helper import generate, load_lora_adapter, load_model_for_inference


if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    lora_adapter_path = "grpo_saved_lora"
    question = "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?"

    # With lora adapter
    model_base, tokenizer = load_model_for_inference(
        model_name=model_name,
        max_seq_length=2048,
        dtype=torch.float16,
        load_in_4bit=True
    )
    # Without lora adapter
    base_response = generate(
        model=model_base, 
        tokenizer=tokenizer, 
        question=question,
        system_prompt=SYSTEM_PROMPT_DETAILED
    )
    print(f"base_response: {base_response}")

    # With lora adapter
    model_peft = load_lora_adapter(model_base, lora_adapter_path=lora_adapter_path, adapter_name="grpo_saved_lora")
    grpo_response = generate(
        model=model_peft, 
        tokenizer=tokenizer, 
        question=question,
        system_prompt=SYSTEM_PROMPT
    )
    print(f"grpo_response: {grpo_response}")