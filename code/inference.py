
from vllm import SamplingParams

def prepare_text(question: str, tokenizer) -> str:
    return tokenizer.apply_chat_template([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        tokenize=False,
        add_generation_prompt=True
    ])

def generate(model, 
             tokenizer, 
             lora_adapter_name: str | None, 
             question: str,
             temperature: float = 0.7,
             top_p: float = 0.95,
             top_k: int = 40,
             max_tokens: int = 1024
             ) -> str:
    text = prepare_text(question=question, tokenizer=tokenizer)
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
    )
    output = model.fast_generate(
        text,
        sampling_params=sampling_params,
        lora_request=model.load_lora(lora_adapter_name) if lora_adapter_name else None,
    )[0].output[0].content
    return output