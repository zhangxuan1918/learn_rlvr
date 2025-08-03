from data import (
    USER_PROMPT_CODE,
    TrainMethodType,
    extract_answer_hash_tag,
    extract_answer_python,
    get_gsm8k_dataset,
)
from inference_helper import generate, load_model_for_inference
from model import load_lora_adapter
import torch
from tqdm import tqdm
from llm_sandbox import SandboxSession


def evaluate_gsm8k(
    model_name: str,
    user_prompt: str,
    lora_adapter_path: str | None = None,
    adapter_name: str = "default",
    batch_size: int = 16,
    python_sandbox_container_id: str | None = None,
) -> float:
    gsm8k = get_gsm8k_dataset(split="test")
    model, tokenizer = load_model_for_inference(
        model_name=model_name,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        max_seq_length=2048,
    )

    if lora_adapter_path:
        model = load_lora_adapter(
            model,
            lora_adapter_path=lora_adapter_path,
            adapter_name=adapter_name,
        )

    correct = 0
    total = 0
    gsm8k_data = list(gsm8k)
    num_batches = (len(gsm8k_data) + batch_size - 1) // batch_size
    with tqdm(range(num_batches), desc="Evaluating") as pbar:
        for batch_idx in pbar:
            examples = gsm8k_data[
                batch_idx * batch_size : batch_idx * batch_size + batch_size
            ]
            questions = [example["question"] for example in examples]
            gt_answers = [example["answer"] for example in examples]
            # Greedy decode
            responses = generate(
                model=model,
                tokenizer=tokenizer,
                questions=questions,
                user_prompt=user_prompt,
                max_new_tokens=1024,
                temperature=None,
                top_p=None,
                top_k=None,
                do_sample=False,
            )
            with SandboxSession(lang="python", container_id=python_sandbox_container_id, verbose=True) as session:
                pred_answers = [
                    extract_answer_python(text=response, session=session)
                    for response in responses
                ]
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
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    run_num = 4
    train_method = TrainMethodType.BASE
    if train_method in [TrainMethodType.SFT, TrainMethodType.GRPO]:
        lora_adapter_path = (
            f"output/{train_method}/{model_name}/run{run_num}/{train_method}_saved_lora"
        )
    else:
        # Base model
        lora_adapter_path = None

    accuracy = evaluate_gsm8k(
        model_name=model_name,
        lora_adapter_path=lora_adapter_path,
        user_prompt=USER_PROMPT_CODE,
        batch_size=64,
        python_sandbox_container_id="a319b3c76c22",
    )
    print(f"\n✅ GSM8K Accuracy ({train_method}): {accuracy:.1%}")
