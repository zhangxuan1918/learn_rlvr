import re
from inference_helper import generate, load_model_for_inference
from model import load_lora_adapter
from data import SYSTEM_FORMAT_PROMPT, SYSTEM_DETAILED_FORMAT_PROMPT

import torch
import torch.nn.functional as F


def get_token_probability(
    model,
    tokenizer,
    question: str,
    response: str,
    system_prompt: str = SYSTEM_FORMAT_PROMPT,
) -> tuple[list[str], torch.Tensor]:
    prompts = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": response},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    # For unknown reason, tokenizer.padding_side is reset to be "right"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        # Remove last token.
        outputs = model(inputs["input_ids"][:, :-1])
        logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(
        2, inputs["input_ids"][:, 1:].unsqueeze(-1)
    ).squeeze(-1)
    token_probs = token_log_probs.exp()
    # Remove first token
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0, 1:])
    return tokens, token_probs


def value_to_bg_color(v: float):
    """
    Map a value in [-1, 1] to a background color:
    -1 → red,  0 → yellow,  1 → green
    """
    # Clamp value to [-1, 1]
    v = max(-1.0, min(1.0, v))
    
    # Normalize to [0, 1]
    norm = (v + 1) / 2  # -1 → 0, 0 → 0.5, 1 → 1

    # Red to Green gradient
    red = int(255 * (1 - norm))
    green = int(255 * norm)
    blue = 0

    return f"\x1b[48;2;{red};{green};{blue}m"  # Background color (truecolor)

def colorize_words_by_value(words: list[str], values: list[float]):
    """
    Print each word with background color based on a value in [-1, 1].
    """
    reset = "\x1b[0m"
    for word, value in zip(words, values):
        color = value_to_bg_color(value)
        print(f"{color} {word} {reset}", end=" ")
    print()

def export_colored_words_to_markdown(words: list[str], values: list[float]):
    """Return a markdown-safe HTML string with colored word spans."""
    def value_to_rgb(value):
        """Convert value in [-1, 1] to (R, G, B) for heatmap effect."""
        value = max(-1.0, min(1.0, value))
        norm = (value + 1) / 2  # -1 → 0, 0 → 0.5, 1 → 1
        red = int(255 * (1 - norm))
        green = int(255 * norm)
        blue = 0
        return red, green, blue
    spans = []
    for word, val in zip(words, values):
        r, g, b = value_to_rgb(val)
        span = f'<span style="background-color: rgb({r},{g},{b}); padding:2px; border-radius:4px; margin:1px;">{word}</span>'
        spans.append(span)
    return " ".join(spans)

def compare_token_probability(ref_model, model, tokenizer, question: str, system_prompt: str, user_prompt: str, max_new_tokens: int = 1024):
    formatted_question = user_prompt.format(question=question)
    # Get peft model response
    response = generate(
        model=model,
        tokenizer=tokenizer,
        questions=[formatted_question],
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_prompt=False,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=True
    )[0]

    # Compute model token prob
    tokens, peft_token_probs = get_token_probability(
        model=model,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        question=formatted_question,
        response=response,
    )
    # Compute ref model token prob
    _, base_token_probs = get_token_probability(
        model=ref_model,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        question=formatted_question,
        response=response,
    )

    cleaned_tokens = [re.sub(r"Ċ", "\n", re.sub(r"Ġ", "", token)) for token in tokens]
    for cleaned_token, peft_prob, base_prob in zip(
        cleaned_tokens, peft_token_probs[0].tolist(), base_token_probs[0].tolist()
    ):
        if cleaned_token:
            print(f"{cleaned_token}: {peft_prob:.4f} vs {base_prob:.4f} (diff: {peft_prob - base_prob:.4f})")
    
    print("-" * 20)
    colorize_words_by_value(
        words=cleaned_tokens,
        values=(peft_token_probs - base_token_probs)[0].tolist(),
    )

    print("-" * 20)
    print(export_colored_words_to_markdown(
        words=cleaned_tokens,
        values=(peft_token_probs - base_token_probs)[0].tolist(),
    ))

if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    run_num = 2
    lora_adapter_path = f"output/grpo/{model_name}/run{run_num}/grpo_saved_lora"

    # question = "Janet buys a brooch for her daughter. She pays $500 for the material to make it and then another $800 for the jeweler to construct it. After that, she pays 10% of that to get it insured. How much did she pay?"
    question = "hours the first day and half as much the other two days he runs. How fast does he run?"
    # Load base model
    base_model, tokenizer = load_model_for_inference(
        model_name=model_name,
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    # Load peft model
    model, _ = load_model_for_inference(
        model_name=model_name,
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    peft_model = load_lora_adapter(
        model_base=model,
        lora_adapter_path=lora_adapter_path,
    )

    compare_token_probability(
        ref_model=base_model,
        model=peft_model,
        tokenizer=tokenizer,
        question=question,
        system_prompt=SYSTEM_FORMAT_PROMPT,
        user_prompt="{question}",
        max_new_tokens=1024,
    )

    compare_token_probability(
        ref_model=peft_model,
        model=base_model,
        tokenizer=tokenizer,
        question=question,
        system_prompt= SYSTEM_DETAILED_FORMAT_PROMPT,
        user_prompt="{question}\nLet's think step by step.",
        max_new_tokens=1024,
    )
