from model import get_model
from data import (
    get_gsm8k_questions,
    xmlcount_reward_func,
    strict_format_reward_func,
    correctness_reward_func,
    int_reward_func,
    soft_format_reward_func,
)
from trl import GRPOConfig, GRPOTrainer


def get_train_config():
    return GRPOConfig(
        use_vllm=True,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adam_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=8,
        max_prompt_length=1024,
        max_completion_length=256,
        num_train_epochs=1,
        save_steps=250,
        max_grad_norm=0.1,
        report_to="none",
        output_dir="output",
    )


def get_trainer(training_config, model, tokenizer, dataset):
    return GRPOTrainer(
        model=model,
        preprocessing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ],
        args=training_config,
        train_dataset=dataset,
    )

def train():
    model, tokenizer = get_model()
    training_config = get_train_config()
    dataset = get_gsm8k_questions()
    trainer = get_trainer(training_config, model, tokenizer, dataset)
    trainer.train()
    model.save_lora("grpo_saved_lora")

if __name__ == "__main__":
    train()