# learn_rlvr

Due to GPU memory, we use QLora to do RLVR. We use Unsloth for QLora and Huggingface TRL for GRPO.

## Qwen/Qwen2.5-3B-Instruct
### Setup
* base model: Qwen/Qwen2.5-3B-Instruct, 4bits
* lora adapter: rank 64, bf16
* KL coef is 0.0
* reward is normalized

### Training

Since we are using QLora and we have small number paramters to train. We don't notice anything abnormal in the training.

![correctness_reward](docs/training/3B_Instruct/train_correctness_reward.png)
![format_reward](docs/training/3B_Instruct/train_strict_format_reward.png)
![completion_length](docs/training/3B_Instruct/train_completion_length.png)
![clipping_ratio](docs/training/3B_Instruct/train_clipping_ratio.png)
### GSM8K Eval

| Model | PASS@1 | Comment |
| ----- | ------ | ------- |
| Base  |69.7%   |         |
| GRPO  |81.1%   |  run1   |
| GRPO  |81.1%   |  run2, small changes in rewards |

Eval setup
* max output token: 1024
* max sequence length: 2048
* greedy decoding
* base model
  * loaded in 4 bits
  * system prompt: 
      ```text
      Respond in the following format:
      <reasoning>
      Put your reasoning here.
      </reasoning>
      <answer>
      Put your answer here. The answer should be an integer.
      </answer>
      ```
  * user prompt:
      ```text
      {question}\nLet's think step by step.
      ```
* lora fine-tuned model
  * base model weights in 4 bits
  * weights in bf16
  * system prompt:
    ```text
    Respond in the following format:
    <reasoning>
    ...
    </reasoning>
    <answer>
    ...
    </answer>
    ```
  * user prompt:
      ```text
      {question}
      ```

The reported GSM8K metric for Qwen 2.5 3B Instruct model is 86.7. The performance difference could due to
1. different prompt: I didn't find the prompt used in Qwen 2.5 report
2. the model is loaded in 4bits
3. instruction following issue: I also saw "many" cases where the base model is able to output the correct answer but the format is wrong, e.g. instead of an integer in the `<answer>` tag, it outputs a sentence or units followed by the answer

### Token Probability

<iframe src="docs/training/3B_instruct/token_probability.html" width="100%" height="100" frameborder="0"></iframe>

### Implementation Issues

1. when using flashinfer together with Unsloth, it complains about Ninja not find when compling flashinfer. Even through the package is installed in the conda enviorment.
   * solution: install Ninja package system wide
   * the compilation is started in a thread. Maybe python path is not properly configured
2. when evaluating GSM8K for base model and GRPOed model, we notice the response in the second batch looks strange: wrong format, repeated tokens etc
   * solution: add ```python tokenizer.padding_side="left"``` for each batch
   * for unknown reason, padding_side is reset to "right" after the first batch inference. [link](https://github.com/unslothai/unsloth/issues/267)

## Qwen/Qwen2.5-0.5B-Instruct
Qwen2.5-0.5B-Instruct doesn't follow the instruction properly. It doesn't put the thought and answer in `<reasoning></reasoning>` and `<answer></answer>` tags. As a result, we first do SFT such that the finetuned model can output in the expected format.

### SFT
We use `unsloth/OpenMathReasoning-mini` to fine tune the base model. To limit the output length, we filter the training data and only keep the data where the sequence length is less than 2048. The training and eval setup are identical as before.

| Model | PASS@1 | Comment |
| ----- | ------ | ------- |
| Base  | 0.0%   | response format issue |
| SFT   | 8.0%   | run1, batch size=64, lr=5e-5 |
| SFT   | 19.6%   | run2, batch size=8, lr=5e-4 |

We notice smaller batch size and larger learning rate gives us a better checkpoint. For run2, one of the loss pattern is the reasoning is lengthy exceeding the max sequence length.
### Training

![correctness_reward](docs/training/0.5B_Instruct/train_correctness_reward.png)
![format_reward](docs/training/0.5B_Instruct/train_strict_format_reward.png)
![completion_length](docs/training/0.5B_Instruct/train_completion_length.png)
![clipping_ratio](docs/training/0.5B_Instruct/train_clipping_ratio.png)

### GSM8K Eval

| Model | PASS@1 | Comment |
| ----- | ------ | ------- |
| Base  | 0.0%   | response format issue |
| SFT   | 8.0%   | run1, batch size=64, lr=5e-5 |
| SFT   | 19.6%   | run2, batch size=8, lr=5e-4 |
| GRPO  | 4.6%   | based on SFT run1 |
| GRPO  | %   | based on SFT run2 |