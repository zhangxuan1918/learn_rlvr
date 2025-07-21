import torch
from tqdm import tqdm

from data import SYSTEM_PROMPT, SYSTEM_PROMPT_DETAILED, extract_xml_answer, get_gsm8k_questions
from inference_helper import generate, load_lora_adapter, load_model_for_inference


def evaluate_gsm8k(
    model_name: str, lora_adapter_path: str | None = None, adapter_name: str = "default"
) -> float:
    gsm8k = get_gsm8k_questions(split="test")
    model, tokenizer = load_model_for_inference(
        model_name=model_name,
        dtype=torch.float16,
        load_in_4bit=True,
        max_seq_length=2048,
    )
    system_prompt = SYSTEM_PROMPT_DETAILED
    if lora_adapter_path:
        model = load_lora_adapter(
            model,
            lora_adapter_path=lora_adapter_path,
            adapter_name=adapter_name,
        )
        system_prompt = SYSTEM_PROMPT

    correct = 0
    total = 0

    for example in tqdm(gsm8k):
        question = example["question"]
        gt_answer = example["answer"].split("####")[-1].strip()
        # Greedy decode
        decoded = generate(
            model=model,
            tokenizer=tokenizer,
            question=question,
            temperature=0.0,
            max_tokens=1024,
            system_prompt=system_prompt,
            do_sample=False,
        )
        # Extract last number after "####" if present
        pred_answer = extract_xml_answer(decoded)

        if pred_answer == gt_answer:
            correct += 1
        total += 1

        # Optional: print a few examples
        if total <= 3:
            print(f"\nQ: {question}")
            print(f"GT: {gt_answer}")
            print(f"PRED: {pred_answer}")

    accuracy = correct / total
    return accuracy


if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    accuracy_base = evaluate_gsm8k(model_name=model_name)
    print(f"\n✅ GSM8K Accuracy (Base): {accuracy_base:.2%}")

    lora_adapter_path = "output/grpo/Qwen/Qwen2.5-3B-Instruct/checkpoint-1750"
    accuracy_grpo = evaluate_gsm8k(model_name=model_name, lora_adapter_path=lora_adapter_path)
    print(f"\n✅ GSM8K Accuracy (GRPO): {accuracy_grpo:.2%}")