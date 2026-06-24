#!/bin/bash
set -ex

MODEL=$1
OUTPUT_ROOT=$2
mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT=$(realpath "$OUTPUT_ROOT")

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
# 强制使用本地缓存，避免 HuggingFace Hub API 429 错误 (需先运行 download_data.py)
export PANGEABENCH_USE_LOCAL_CACHE=1

# TASKS comes from env var `TASKS` (comma-separated) if provided; otherwise use defaults.
# Example:
#   export TASKS="maxm,xmmmu,cvqa,marvl,xm100,xgqa,m3exam"
DEFAULT_TASKS=('maxm' 'cvqa' 'xgqa' 'xmmmu' 'xm100' 'marvl' 'm3exam')
if [ -n "${TASKS:-}" ]; then
    IFS=',' read -r -a TASKS_ARR <<< "${TASKS}"
else
    TASKS_ARR=("${DEFAULT_TASKS[@]}")
fi

export WORKERS=512 # preprocess workers, exps/mmlm_via_merge/src/Pangea/evaluation/lmms-eval/lmms_eval/models/vllm.py#L24

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=$(nvidia-smi -L | wc -l)
else
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi

echo "Model: $MODEL"
echo "Output Directory: $OUTPUT_ROOT"
echo "Tasks: ${TASKS_ARR[*]}"
echo "Number of GPUs: $NUM_GPUS"

source uv_pangea/bin/activate
cd src/Pangea/evaluation/lmms-eval

# for iteration in {1..10}; do
#     echo "========================================="
#     echo "Iteration $iteration of 10"
#     echo "========================================="

for task in "${TASKS_ARR[@]}"; do
    echo "Evaluating task: $task"
    echo "Output path: ${OUTPUT_ROOT}/${MODEL}/${task}"
    LIMIT_ARGS=""
    if [ "$task" = "cvqa" ]; then
        LIMIT_ARGS="--limit 500"
    elif [ "$task" = "marvl" ]; then
        LIMIT_ARGS="--limit 100"    # 100 per subset, 600 out of 12652 in total
    elif [ "$task" = "xgqa" ]; then
        LIMIT_ARGS="--limit 1000"    # 1000 per subset, 8000 out of 77328 in total
    fi
    if echo "$MODEL" | grep -qi "onevision"; then
        python3 -m lmms_eval \
               --model llava_onevision1_5 \
               --model_args pretrained="$MODEL" \
               --tasks "${task}" \
               ${LIMIT_ARGS} \
               --batch_size 1 \
               --log_samples \
               --log_samples_suffix "${task}" \
               --output_path "${OUTPUT_ROOT}/${MODEL}/${task}"
               # --overwrite
    elif echo "$MODEL" | grep -qi "pangea"; then
        python3 -m accelerate.commands.launch \
                --num_processes=${NUM_GPUS} \
                -m lmms_eval \
                --model pangea \
                --model_args pretrained=${MODEL} \
                --tasks ${task} \
                ${LIMIT_ARGS} \
                --batch_size 1 \
                --log_samples \
                --log_samples_suffix ${task} \
                --output_path "${OUTPUT_ROOT}/${MODEL}/${task}"
                # --overwrite
    else
        python3 -m lmms_eval \
            --model vllm \
            --model_args model="${MODEL}",tensor_parallel_size="${NUM_GPUS}" \
            --tasks "${task}" \
            ${LIMIT_ARGS} \
            --batch_size 8192 \
            --log_samples \
            --log_samples_suffix "${task}" \
            --output_path "${OUTPUT_ROOT}/${MODEL}/${task}" \
            --gen_kwargs "temperature=0,do_sample=False,top_p=1,seed=42"
            # --overwrite
    fi
    sleep 20
done
#     echo "Completed iteration $iteration of 5"
#     echo ""
#     sleep 120
# done


# export TASKS="maxm,xmmmu,cvqa,marvl,xm100,xgqa,m3exam"
# MODEL=/path/to/Qwen2.5-VL-7B-Instruct
# OUTPUT_ROOT=outputs/eval_pangeabench
# ts -G 1 bash scripts/eval/pangeabench/eval_pangeabench.sh "$MODEL" "$OUTPUT_ROOT"

# export TASKS="multilingual_llava_bench,xchat"
# MODEL=/path/to/Qwen2.5-VL-7B-Instruct
# OUTPUT_ROOT=outputs/eval_pangeabench
# ts -G 1 bash scripts/eval/pangeabench/eval_pangeabench.sh "$MODEL" "$OUTPUT_ROOT"
