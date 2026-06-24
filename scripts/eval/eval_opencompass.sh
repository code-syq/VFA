#!/bin/bash
# VFA Evaluation Script - Text-only Multilingual Benchmarks via OpenCompass
# Usage: bash scripts/eval/eval_opencompass.sh <MODEL_PATH> <OUTPUT_ROOT> [chat|base] [TASKS]
#
# Environment variables:
#   TASKS: comma-separated task list
#   MAX_SEQ_LEN: max sequence length (default: 4096)
#   GENERATION_KWARGS: generation parameters (default: temperature=0 top_p=1 top_k=-1 do_sample=False seed=42)
set -ex

# Activate opencompass environment
if [ -f "uv_opencompass/bin/activate" ]; then
    source uv_opencompass/bin/activate
fi

MODEL=$1
OUTPUT_ROOT=$2
CHAT_TEMPLATE=${3:-"chat"}
TASKS_CLI=${4:-""}
MAX_OUT_LEN=2048
MAX_SEQ_LEN=${MAX_SEQ_LEN:-4096}

if [ -z "${MODEL:-}" ] || [ -z "${OUTPUT_ROOT:-}" ]; then
    echo "Usage: bash scripts/eval/eval_opencompass.sh <MODEL> <OUTPUT_ROOT> [chat|base] [TASKS]"
    echo "  Example TASKS: tydiqa_gen,mmmlu_gen_d5017d,xnli_gen_973734,mhellaswag_gen_1a6b73,mifeval_gen_79f8fb,mlogiqa_gen_36c4f9,flores_gen_2697d7"
    exit 1
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=0
fi

GENERATION_KWARGS=${GENERATION_KWARGS:-"temperature=0 top_p=1 top_k=-1 do_sample=False seed=42"}

mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT=$(realpath "$OUTPUT_ROOT")

# Default tasks: multilingual text-only benchmarks
DEFAULT_TASKS=('tydiqa_gen' 'mmmlu_gen_d5017d' 'xnli_gen_973734' 'mhellaswag_gen_1a6b73' 'mifeval_gen_79f8fb' 'mlogiqa_gen_36c4f9' 'flores_gen_2697d7')

if [ -n "${TASKS:-}" ]; then
    IFS=',' read -r -a TASKS_ARR <<< "${TASKS}"
elif [ -n "${TASKS_CLI}" ]; then
    IFS=',' read -r -a TASKS_ARR <<< "${TASKS_CLI}"
else
    TASKS_ARR=("${DEFAULT_TASKS[@]}")
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NUM_GPUS=$(nvidia-smi -L | wc -l)
else
    NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
fi

echo "Model: $MODEL"
echo "Output Root: $OUTPUT_ROOT"
echo "Chat Template: $CHAT_TEMPLATE"
echo "Tasks: ${TASKS_ARR[*]}"
echo "Number of GPUs: $NUM_GPUS"

export VLLM_WORKER_MULTIPROC_METHOD=spawn

for task in "${TASKS_ARR[@]}"; do
    OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL}/${task}"

    echo "========================================"
    echo "Evaluating task: $task"
    echo "Output Directory: $OUTPUT_DIR"

    if echo "$MODEL" | grep -qi "onevision"; then
        opencompass \
        --datasets "${task}" \
        --work-dir "$OUTPUT_DIR" \
        --debug \
        --hf-type "$CHAT_TEMPLATE" \
        --hf-path "$MODEL" \
        --max-seq-len "${MAX_SEQ_LEN}" \
        --max-out-len "${MAX_OUT_LEN}" \
        --hf-num-gpus "${NUM_GPUS}" \
        --batch-size 16 \
        --generation-kwargs ${GENERATION_KWARGS}
    else
        opencompass \
        --datasets "${task}" \
        --work-dir "$OUTPUT_DIR" \
        --debug \
        --accelerator vllm \
        --hf-type "$CHAT_TEMPLATE" \
        --hf-path "$MODEL" \
        --max-seq-len "${MAX_SEQ_LEN}" \
        --max-out-len "${MAX_OUT_LEN}" \
        --hf-num-gpus "${NUM_GPUS}" \
        --batch-size 1024 \
        --model-kwargs tensor_parallel_size=${NUM_GPUS} gpu_memory_utilization=0.9 \
        --generation-kwargs ${GENERATION_KWARGS}
    fi
done
