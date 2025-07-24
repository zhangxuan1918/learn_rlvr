import re
from datasets import load_dataset

SYSTEM_PROMPT = """Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>"""

SYSTEM_PROMPT_DETAILED = """Respond in the following format:
<reasoning>
Put your reasoning here.
</reasoning>
<answer>
Put your answer here. The answer should be a numeric value.
</answer>"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>"""

# Loads dataset helper functions
def get_gsm8k_dataset(split = "train"):
    data = load_dataset("openai/gsm8k", "main", split=split)
    data = data.map(lambda x: {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": x["question"]},
        ],
        "answer": extract_hash_answer(x["answer"])
    })
    return data

def get_open_math_reasoning_dataset(split = "cot"):
    data = load_dataset("unsloth/OpenMathReasoning-mini", split=split)
    data = data.select_columns(["problem", "generated_solution", "expected_answer"])
    # Filter out prompts with non-numeric expected answer
    data = data.filter(lambda example: example["expected_answer"].isnumeric())
    
    # Format dataset
    def format_data(x):
        expected_answer = x["expected_answer"].strip()
        problem = x["problem"].strip()
        # Replace <think> by <reasoning>, </think> by </reasoning>
        thoughts = x["generated_solution"]
        thoughts = thoughts.replace("<think>", "").replace("</think>", "").strip()
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": problem},
                {"role": "assistant", "content": XML_COT_FORMAT.format(reasoning=thoughts, answer=expected_answer)},
            ],
            "answer": expected_answer,
            "thoughts": thoughts,
        }
    data = data.map(format_data)
    return data

# Extract answer helper functions
def extract_xml_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)

    if match:
        return match.group(1).strip()
    else:
        return ""

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    question = prompts[0][-1]["content"]
    extracted_responses = [extract_xml_answer(response) for response in responses]
    print("-" * 20, f"Question:\n{question}", f"\nAnswer:\n{answer[0]}", f"\nResponses:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(response) for response in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.75 if m else 0.0 for m in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.25 if m else 0.0 for m in matches]

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
