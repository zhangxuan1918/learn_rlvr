from model import get_model
from data import get_open_math_reasoning_dataset

import os
from trl import SFTConfig, SFTTrainer
import wandb
from dotenv import load_dotenv

load_dotenv()


def get_train_config(
    input_name: str = "text", report_to: str = "none", output_dir: str = "output"
):
    return SFTConfig(
        dataset_text_field=input_name,
        learning_rate=2e-5,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=5,
        per_device_train_batch_size=64,
        gradient_accumulation_steps=1,
        num_train_epochs=2,
        report_to=report_to,
        output_dir=output_dir,
    )


def get_trainer(training_config, model, tokenizer, dataset):
    return SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_config,
        train_dataset=dataset,
    )


def preprocess_dataset(dataset, tokenizer, max_seq_length: int, input_name="text"):
    def _apply_chat_template(example):
        example[input_name] = tokenizer.apply_chat_template(
            example["prompt"], tokenize=False
        )
        return example

    return dataset.map(_apply_chat_template).filter(
        lambda example: len(tokenizer(example[input_name])["input_ids"]) <= max_seq_length
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
):
    model, tokenizer = get_model(
        model_name=model_name,
        lora_rank=lora_rank,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=fast_inference,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    input_name = "text"
    dataset = preprocess_dataset(
        dataset=get_open_math_reasoning_dataset(),
        tokenizer=tokenizer,
        input_name=input_name,
        max_seq_length=max_seq_length,
    )

    training_config = get_train_config(
        input_name=input_name, report_to=report_to, output_dir=output_dir
    )
    trainer = get_trainer(
        training_config=training_config,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
    )
    trainer.train()
    model.save_pretrained(os.path.join(output_dir, "sft_saved_lora"))


if __name__ == "__main__":

    run_num = 3
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    output_dir = f"output/sft/{model_name}/run{run_num}"

    if os.environ.get("WANDB_API_KEY", None):
        wandb.login(key=os.environ["WANDB_API_KEY"])
        wandb.init(
            project="learn-rlvr",
            config=get_train_config().to_dict(),
            name=f"sft_{model_name}_run{run_num}",
        )
        report_to = "wandb"
    else:
        report_to = "none"

    train(
        model_name=model_name,
        max_seq_length=2048,
        load_in_4bit=True,
        fast_inference=False,
        lora_rank=64,
        gpu_memory_utilization=0.7,
        report_to=report_to,
        output_dir=output_dir,
    )
    wandb.finish()
