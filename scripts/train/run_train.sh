#!/bin/bash
# VFA Training Script - Fine-tune LLM/VLM via LLaMA-Factory
# Usage: bash scripts/train/run_train.sh <CONFIG_YAML> [NUM_GPUS]
#
# Examples:
#   bash scripts/train/run_train.sh configs/finetune_llm/Qwen2.5-7B/multilingual_sft_99999.yaml
#   bash scripts/train/run_train.sh configs/finetune_vlm/Qwen2.5-VL-7B-Instruct/multilingual_sft_99999.yaml 4
set -ex

CONFIG=$1
NUM_GPUS=${2:-$(nvidia-smi -L 2>/dev/null | wc -l)}

if [ -z "$CONFIG" ]; then
    echo "Usage: bash scripts/train/run_train.sh <CONFIG_YAML> [NUM_GPUS]"
    echo ""
    echo "Available configs:"
    find configs/finetune_llm configs/finetune_vlm -name "*.yaml" 2>/dev/null | sort
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

source uv_llamafactory/bin/activate

echo "========================================"
echo "Training config: $CONFIG"
echo "Number of GPUs: $NUM_GPUS"
echo "========================================"

# Multi-GPU via torchrun; single-GPU direct
export FORCE_TORCHRUN=1
export NNODES=1
export NPROC_PER_NODE=$NUM_GPUS

llamafactory-cli train "$CONFIG" 2>&1 | tee "train_$(basename $CONFIG .yaml).log"

echo "Training completed!"
