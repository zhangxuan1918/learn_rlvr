from data import SYSTEM_PROMPT

from unsloth import FastLanguageModel
from peft import PeftModel
import torch


def generate(
    model,
    tokenizer,
    questions: list[str],
    temperature: float | None = 0.7,
    top_p: float | None = 0.95,
    top_k: int | None = 40,
    max_new_tokens: int = 512,
    do_sample: bool = True,
    system_prompt: str = SYSTEM_PROMPT,
    return_prompt: bool = False
) -> list[str]:
    prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for question in questions
    ]
    # For unknown reason, tokenizer.padding_side is reset to be "right"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            do_sample=do_sample,  # Greedy decode if do_sample=False
        )
    if not return_prompt:
        generated_tokens = outputs[:, inputs["input_ids"].shape[1] :]
    else:
        generated_tokens = outputs
    generated_texts = tokenizer.batch_decode(
        generated_tokens, 
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    return generated_texts


def load_model_for_inference(
    model_name: str, max_seq_length: int, dtype: torch.dtype, load_in_4bit: bool
):
    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    model.eval()
    FastLanguageModel.for_inference(model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


def load_lora_adapter(
    model_base, lora_adapter_path: str, adapter_name: str = "default"
):
    return PeftModel.from_pretrained(
        model=model_base,
        model_id=lora_adapter_path,
        adapter_name=adapter_name,
    )
