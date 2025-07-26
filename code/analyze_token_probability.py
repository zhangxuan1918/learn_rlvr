import re
from inference_helper import generate, load_model_for_inference
from model import load_lora_adapter
from data import SYSTEM_PROMPT

import torch
import torch.nn.functional as F


def get_token_probability(
    model,
    tokenizer,
    question: str,
    response: str,
    system_prompt: str = SYSTEM_PROMPT,
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

def compare_token_probability(base_model, peft_model, tokenizer, question: str):
    # Get peft model response
    response = generate(
        model=peft_model,
        tokenizer=tokenizer,
        questions=[question],
        system_prompt=SYSTEM_PROMPT,
        do_sample=False,
        return_prompt=False,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False
    )[0]

    # Compute peft model token prob
    tokens, peft_token_probs = get_token_probability(
        model=peft_model,
        tokenizer=tokenizer,
        question=question,
        response=response,
    )
    # Compute base model token prob
    _, base_token_probs = get_token_probability(
        model=base_model,
        tokenizer=tokenizer,
        question=question,
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
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    run_num = 1
    lora_adapter_path = f"output/grpo/{model_name}/run{run_num}/grpo_saved_lora"

    question = "Gloria is shoe shopping when she comes across a pair of boots that fit her shoe budget. However, she has to choose between the boots and two pairs of high heels that together cost five dollars less than the boots. If one pair of heels costs $33 and the other costs twice as much, how many dollars are the boots?"

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
        base_model=base_model,
        peft_model=peft_model,
        tokenizer=tokenizer,
        question=question,
    )
