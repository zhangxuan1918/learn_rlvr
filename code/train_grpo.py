from typing import Callable
from llm_sandbox import SandboxSession
from model import get_model
from data import (
    OutputType,
    get_gsm8k_dataset,
    random_reward_func,
    xmlcount_reward_func,
    strict_format_reward_func,
    correctness_reward_func,
    int_reward_func,
    soft_format_reward_func,
    get_code_reward_func,
)
import os
from trl import GRPOConfig, GRPOTrainer
import wandb
from dotenv import load_dotenv

load_dotenv()


def get_train_config(report_to: str = "none", output_dir: str = "output"):
    return GRPOConfig(
        use_vllm=True,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        num_generations=8,
        max_prompt_length=1024,
        max_completion_length=1024,
        num_train_epochs=1,
        save_steps=250,
        max_grad_norm=0.1,
        report_to=report_to,
        output_dir=output_dir,
    )


def get_trainer(training_config, model, tokenizer, dataset, reward_funcs):
    return GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_config,
        train_dataset=dataset,
    )


def train(
    model_name: str,
    max_seq_length: int,
    load_in_4bit: bool,
    fast_inference: bool,
    lora_rank: int,
    gpu_memory_utilization: float,
    report_to: str,
    output_dir: str,
    lora_adapter_path: str | None = None,
    reward_funcs: list = [correctness_reward_func],
):
    model, tokenizer = get_model(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=fast_inference,
        lora_rank=lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,
        lora_adapter_path=lora_adapter_path,
    )

    training_config = get_train_config(report_to=report_to, output_dir=output_dir)
    dataset = get_gsm8k_dataset()

    trainer = get_trainer(training_config, model, tokenizer, dataset, reward_funcs)
    trainer.train()
    model.save_pretrained(os.path.join(output_dir, "grpo_saved_lora"))


def train_grpo_python_code(
    model_name: str, output_dir: str, lora_adapter_path=None, report_to="none"
):
    with SandboxSession(language="python") as session:
        train(
            model_name=model_name,
            max_seq_length=2048,
            load_in_4bit=True,
            fast_inference=True,
            lora_rank=64,
            gpu_memory_utilization=0.7,
            report_to=report_to,
            output_dir=output_dir,
            lora_adapter_path=lora_adapter_path,
            reward_funcs=[
                random_reward_func,
                xmlcount_reward_func,
                strict_format_reward_func,
                int_reward_func,
                soft_format_reward_func,
                get_code_reward_func(session=session),
            ],
        )


def train_grpo_xml(
    model_name: str, output_dir: str, lora_adapter_path=None, report_to="none"
):
    train(
        model_name=model_name,
        max_seq_length=2048,
        load_in_4bit=True,
        fast_inference=True,
        lora_rank=64,
        gpu_memory_utilization=0.7,
        report_to=report_to,
        output_dir=output_dir,
        lora_adapter_path=lora_adapter_path,
        reward_funcs=[
            random_reward_func,
            xmlcount_reward_func,
            strict_format_reward_func,
            int_reward_func,
            soft_format_reward_func,
            correctness_reward_func,
        ],
    )


if __name__ == "__main__":
    # model_name = "meta-llama/Llama-3.2-3B-Instruct"
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    output_type = OutputType.CODE
    run_num = 4
    output_dir = f"output/grpo/{model_name}/run{run_num}"
    # If we first sft the model, we need to load the lora adapter
    # lora_adapter_path = f"output/sft/{model_name}/run5/sft_saved_lora"
    lora_adapter_path = None
    if os.environ.get("WANDB_API_KEY", None):
        wandb.login(key=os.environ["WANDB_API_KEY"])
        wandb.init(
            project="learn-rlvr",
            config=get_train_config().to_dict(),
            name=f"grpo_{model_name}_run{run_num}",
        )
        report_to = "wandb"
    else:
        report_to = "none"

    match output_type:
        case OutputType.XML:
            train_grpo_xml(
                model_name=model_name,
                output_dir=output_dir,
                lora_adapter_path=lora_adapter_path,
                report_to=report_to,
            )
        case OutputType.CODE:
            train_grpo_python_code(
                model_name=model_name,
                output_dir=output_dir,
                lora_adapter_path=lora_adapter_path,
                report_to=report_to,
            )
    wandb.finish()
