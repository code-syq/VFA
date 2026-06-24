#!/bin/bash
# Merge LoRA adapter into base model using LLaMA-Factory
# Usage: bash scripts/utils/merge_lora.sh <BASE_MODEL> <LORA_CKPT> [OUTPUT_DIR]
set -e

source uv_llamafactory/bin/activate
BASE_MODEL=$1
LORA_CKPT=$2
OUTPUT_DIR=$3

if [ -z "$BASE_MODEL" ] || [ -z "$LORA_CKPT" ]; then
    echo "Usage: bash scripts/utils/merge_lora.sh <BASE_MODEL> <LORA_CKPT> [OUTPUT_DIR]"
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="${LORA_CKPT}/huggingface"
fi

llamafactory-cli export \
    --model_name_or_path $BASE_MODEL \
    --adapter_name_or_path $LORA_CKPT \
    --export_dir $OUTPUT_DIR \
    --export_size 2 \
    --trust_remote_code=True \
    --export_legacy_format False

echo "Merged LoRA model saved to: $OUTPUT_DIR"
