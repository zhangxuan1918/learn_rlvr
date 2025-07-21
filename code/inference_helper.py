from unsloth import FastLanguageModel
from peft import PeftModel
import torch
from data import SYSTEM_PROMPT

def generate(model, 
             tokenizer, 
             question: str,
             temperature: float = 0.7,
             top_p: float = 0.95,
             top_k: int = 40,
             max_tokens: int = 1024,
             do_sample: bool = True,
             system_prompt: str = SYSTEM_PROMPT
             ):
    input_ids = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        tokenize=True,
        padding=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    if not do_sample:
        # Greedy decoding
        outputs = model.generate(
            input_ids=input_ids.to(model.device),
            attention_mask=(input_ids != tokenizer.pad_token_id).long().to(model.device),
            temperature=1.0,  # Not used
            max_new_tokens=max_tokens,
            do_sample=False,
        )
    else:
        outputs = model.generate(
            input_ids=input_ids.to(model.device),
            attention_mask=(input_ids != tokenizer.pad_token_id).long().to(model.device),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_tokens,
            do_sample=True,
        )
    generated_tokens = outputs[0][input_ids.shape[-1]:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_text

def load_model_for_inference(model_name: str, max_seq_length: int, dtype: torch.dtype, load_in_4bit: bool):
    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def load_lora_adapter(model_base, lora_adapter_path: str, adapter_name: str = "default"):
    return PeftModel.from_pretrained(
        model=model_base,
        model_id=lora_adapter_path,
        adapter_name=adapter_name,
    )