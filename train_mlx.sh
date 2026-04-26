#!/usr/bin/env bash
# Run LoRA fine-tuning on Qwen2.5-7B-Instruct-4bit via MLX.
#
# Requires:
#   pip install mlx-lm
#   data files in data/{train,valid,test}.jsonl  (run split.py first)
#
# After:
#   adapters/                  — trained LoRA weights
#   adapters/adapter_config.json + *.safetensors
#
# To merge the adapter into a fused model for deployment:
#   mlx_lm.fuse \
#     --model mlx-community/Qwen2.5-7B-Instruct-4bit \
#     --adapter-path adapters \
#     --save-path fused-model
set -euo pipefail
cd "$(dirname "$0")"

mlx_lm.lora -c train_config.yaml --train "$@"
