from functools import partial
from typing import Callable

from llm_sandbox import SandboxSession
from data import (
    SYSTEM_FORMAT_PROMPT,
    SYSTEM_DETAILED_FORMAT_PROMPT,
    USER_SIMPLE_PROMPT,
    USER_CODE_PROMPT,
    USER_REASONING_PROMPT,
    OutputType,
    TrainMethodType,
    extract_python_code_answer,
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
    system_prompt: str = "",
    user_prompt=USER_REASONING_PROMPT,
    extract_answer_fn: Callable = extract_xml_answer,
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
            pred_answers = [extract_answer_fn(response) for response in responses]
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


def evaluate_gsm8k_python_code(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    train_method: TrainMethodType,
    lora_adapter_path: str | None = None,
):
    """
    Model generates python code to solve the problem
    """

    with SandboxSession(language="python") as session:
        extract_answer_fn = partial(extract_python_code_answer, session=session)
        accuracy = evaluate_gsm8k(
            model_name=model_name,
            lora_adapter_path=lora_adapter_path,
            batch_size=64,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            extract_answer_fn=extract_answer_fn,
        )
        print(f"\n✅ GSM8K Accuracy ({train_method.name}): {accuracy:.1%}")


def evaluate_gsm8k_xml(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    train_method: TrainMethodType,
    lora_adapter_path: str | None = None,
):
    """
    Model generates answer in xml tags to solve the problem
    """
    accuracy = evaluate_gsm8k(
        model_name=model_name,
        lora_adapter_path=lora_adapter_path,
        batch_size=64,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        extract_answer_fn=extract_xml_answer,
    )
    print(f"\n✅ GSM8K Accuracy ({train_method.name}): {accuracy:.1%}")


if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    run_num = 4
    train_method = TrainMethodType.GRPO
    output_type = OutputType.CODE
    if train_method in ["sft", "grpo"]:
        lora_adapter_path = (
            f"output/{train_method}/{model_name}/run{run_num}/{train_method}_saved_lora"
        )
    else:
        # Base model
        lora_adapter_path = None

    match output_type:
        case OutputType.XML:
            match train_method:
                case TrainMethodType.GRPO:
                    system_prompt = SYSTEM_FORMAT_PROMPT
                    user_prompt = USER_SIMPLE_PROMPT
                case TrainMethodType.SFT, TrainMethodType.BASE:
                    system_prompt = SYSTEM_DETAILED_FORMAT_PROMPT
                    user_prompt = USER_REASONING_PROMPT
                case _:
                    raise ValueError(f"Unknown train method: {train_method}")
            # Evaluate reasoning and answer
            evaluate_gsm8k_xml(
                model_name=model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                lora_adapter_path=lora_adapter_path,
                train_method=train_method,
            )
        case OutputType.CODE:
            # Evaluate Python code
            evaluate_gsm8k_python_code(
                model_name=model_name,
                system_prompt="",
                user_prompt=USER_CODE_PROMPT,
                lora_adapter_path=lora_adapter_path,
                train_method=train_method,
            )
        case _:
            raise ValueError(f"Unknown output type: {output_type}")