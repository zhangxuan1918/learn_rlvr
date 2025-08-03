import enum
import random
import re
from typing import Callable
from datasets import load_dataset

XML_COT_TEMPLATE = """<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>"""

USER_PROMPT_XML_COT = """Answer the question in the following format:
<reasoning>
Put your reasoning here.
</reasoning>
<answer>
Put your answer here. The answer should be an integer.
</answer>
Question: {question}
Let's think step by step!
"""

USER_PROMPT_CODE = """Write a python function to solve the question. For the python function
1. You are not allowed to import any other library
2. Your python function shouldn't take any argument
3. Your python function should only return an integer

Question: {question}
Let's think step by step!
"""


class TrainMethodType(enum.Enum):
    # Base model
    BASE = 0
    SFT = 1
    GRPO = 2


# Loads dataset helper functions
def get_gsm8k_dataset(split="train"):
    data = load_dataset("openai/gsm8k", "main", split=split)
    data = data.map(
        lambda x: {
            "question": x["question"],
            "answer": extract_answer_hash_tag(x["answer"]),
        }
    )
    return data


def get_open_math_reasoning_dataset(split="cot"):
    data = load_dataset("unsloth/OpenMathReasoning-mini", split=split)
    data = data.select_columns(["problem", "generated_solution", "expected_answer"])
    # Filter out prompts with non-numeric expected answer
    data = data.filter(lambda example: example["expected_answer"].isnumeric())

    # Format dataset
    def format_data(x):
        expected_answer = x["expected_answer"].strip()
        problem = x["problem"].strip()
        thoughts = x["generated_solution"]
        thoughts = thoughts.replace("<think>", "").replace("</think>", "").strip()
        return {
            # "prompt": [
            #     {"role": "user", "content": user_prompt.format(question=problem)},
            #     {
            #         "role": "assistant",
            #         "content": content_template.format(
            #             reasoning=thoughts, answer=expected_answer
            #         ),
            #     },
            # ],
            "question": problem,
            "answer": expected_answer,
            "thoughts": thoughts,
        }

    data = data.map(format_data)
    return data


# Extract answer helper functions
def extract_answer_xml(text: str, **kwargs) -> str | None:
    del kwargs
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)

    return postprocess_executed_result(executed_result=matches[-1]) if matches else None


def extract_answer_hash_tag(text: str, **kwargs) -> str | None:
    del kwargs
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def extract_python_code(text: str) -> str | None:
    # Find the last python code block
    pattern = r"```python\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[-1].strip() if matches else None


def execute_python_code_in_sandbox(code: str | None, session) -> str | None:
    if code is None:
        return None
    for i in range(3):
        try:
            return session.run(code, timeout=10).stdout.strip()
        except Exception as e:
            print(f"[{i}/3] Error executing code in sandbox: {e}, retrying...")
            session.open()
    return None


def extract_answer_python(text: str, session) -> str | None:
    try:
        executed_result = execute_python_code_in_sandbox(
            code=extract_python_code(text=text), session=session
        )
        return postprocess_executed_result(executed_result=executed_result)
    except Exception as e:
        return None


def postprocess_executed_result(executed_result: str | None) -> str | None:
    if executed_result is None:
        return None
    matches = re.findall(r"-?\d+\.?\d*", executed_result)
    return matches[-1] if matches else None


# Reward functions
def compute_correctness_reward(
    extracted_answer: str | None, ground_truth_answer: str
) -> float:
    if extracted_answer is None:
        return 0.0
    converted_answer = str(int(float(extracted_answer)))
    if converted_answer == ground_truth_answer:
        return 2.0
    else:
        # extracted answer is a number but not equal to ground truth
        return 0.5


def get_correctness_reward_func(context_manager, extract_answer_fn: Callable):
    def reward_func(prompts, completions, answer, **kwargs) -> list[float]:
        responses = [completion[0]["content"] for completion in completions]
        question = prompts[0][-1]["content"]

        with context_manager() as session:
            executed_answer = [
                extract_answer_fn(response, session=session) for response in responses
            ]
        print(
            "-" * 20,
            f"Question:\n{question}",
            f"\nAnswer:\n{answer[0]}",
            f"\nResponses:\n{responses[0]}",
            f"\nExtracted:\n{executed_answer[0]}",
        )
        return [
            compute_correctness_reward(r, a) for r, a in zip(executed_answer, answer)
        ]

    return reward_func


def random_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    del prompts, answer, kwargs
    return [1.0 if random.random() >= 0.5 else 0.0 for _ in completions]


def format_reward_func(completions, **kwargs) -> list[float]:
    del kwargs
    strict_format_pattern = (
        r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>$"
    )
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(strict_format_pattern, r, flags=re.DOTALL) for r in responses]
    strict_format_rewards = [0.75 if m else 0.0 for m in matches]

    soft_format_pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    matches = [re.match(soft_format_pattern, r, flags=re.DOTALL) for r in responses]
    soft_format_rewards = [0.25 if m else 0.0 for m in matches]

    return [
        max(strict_reward, soft_reward)
        for strict_reward, soft_reward in zip(
            strict_format_rewards, soft_format_rewards
        )
    ]


def count_xml(text: str) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.1
    if text.count("\n</reasoning>\n") == 1:
        count += 0.1
    if text.count("\n<answer>\n") == 1:
        count += 0.1
    if text.count("\n</answer>\n") == 1:
        count += 0.1
        count -= (len(text.split("\n</answer>\n")[-1])) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    return [count_xml(r) for r in responses]
