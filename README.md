# learn_rlvr

## Qwen/Qwen2.5-3B-Instruct
### Setup

Due to GPU memory, we use QLora to do RLVR.
* QLora:
  * base model: Qwen/Qwen2.5-3B-Instruct, 4bits
  * lora adapter: rank 64, bf16

## GSM8K Eval

Eval setup
* max output token: 1024
* greedy decoding
* base model loaded in 4bits
* lora adapter weights in bf16

| Model | PASS@1 |
| ----- | --------------- |
| Base  ||
| post-trained (GRPO) ||