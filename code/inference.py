from data import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_DETAILED,
    extract_xml_answer,
    get_gsm8k_dataset,
)
from inference_helper import generate, load_model_for_inference
from model import load_lora_adapter
import torch
from tqdm import tqdm


def evaluate_gsm8k(
    model_name: str,
    lora_adapter_path: str | None = None,
    adapter_name: str = "default",
    batch_size=16,
) -> float:
    gsm8k = get_gsm8k_dataset(split="test")
    model, tokenizer = load_model_for_inference(
        model_name=model_name,
        dtype=torch.float16,
        load_in_4bit=True,
        max_seq_length=2048,
    )
    system_prompt = SYSTEM_PROMPT_DETAILED
    user_prompt = "{question}\nLet's think step by step."
    if lora_adapter_path:
        model = load_lora_adapter(
            model,
            lora_adapter_path=lora_adapter_path,
            adapter_name=adapter_name,
        )
        system_prompt = SYSTEM_PROMPT
        user_prompt = "{question}"

    correct = 0
    total = 0
    gsm8k_data = list(gsm8k)
    num_batches = (len(gsm8k_data) + batch_size - 1) // batch_size
    with tqdm(range(num_batches), desc="Evaluating") as pbar:
        for batch_idx in pbar:
            examples = gsm8k_data[
                batch_idx * batch_size : batch_idx * batch_size + batch_size
            ]
            questions = [
                user_prompt.format(question=example["question"]) for example in examples
            ]
            gt_answers = [
                example["answer"].split("####")[-1].strip() for example in examples
            ]
            # Greedy decode
            responses = generate(
                model=model,
                tokenizer=tokenizer,
                questions=questions,
                max_new_tokens=1024,
                temperature=None,
                top_p=None,
                top_k=None,
                system_prompt=system_prompt,
                do_sample=False,
            )
            pred_answers = [extract_xml_answer(response) for response in responses]
            batch_correctness = [
                pred_anwser == gt_answer
                for pred_anwser, gt_answer in zip(pred_answers, gt_answers)
            ]
            correct += sum(batch_correctness)
            total += len(batch_correctness)

            pbar.set_postfix(
                {
                    "correct": f"{correct}",
                    "total": f"{total}",
                    "accuracy": f"{correct / total:.4f}",
                }
            )

    accuracy = correct / total
    return accuracy


if __name__ == "__main__":
    run_num = 5
    train_method = "sft"
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    if train_method in ["sft", "grpo"]:
        lora_adapter_path = f"output/{train_method}/{model_name}/run{run_num}/{train_method}_saved_lora"
    else:
        lora_adapter_path = None
    accuracy = evaluate_gsm8k(
        model_name=model_name, lora_adapter_path=lora_adapter_path, batch_size=256
    )
    print(f"\n✅ GSM8K Accuracy ({train_method}): {accuracy:.1%}")
