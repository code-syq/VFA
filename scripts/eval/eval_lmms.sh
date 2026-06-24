#!/bin/bash
set -ex

MODEL=$1
OUTPUT_ROOT=$2
mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT=$(realpath "$OUTPUT_ROOT")

# export HF_HOME="$HOME/.cache/huggingface"
# export HF_HUB_CACHE="$HF_HOME/hub"
# export HF_DATASETS_CACHE="$HF_HOME/datasets"
# export TRANSFORMERS_CACHE="$HF_HOME/transformers"
# 强制使用本地缓存，避免 HuggingFace Hub API 429 错误 (需先运行 download_data.py)
# export PANGEABENCH_USE_LOCAL_CACHE=1

export API_TYPE="azure_msra"
export MODEL_VERSION="gpt-4o"

# TASKS comes from env var `TASKS` (comma-separated) if provided; otherwise use defaults.
# Example:
#   export TASKS="ocrbench_v2,mmbench_en_dev,mmmu_val,mathvista_testmini"
DEFAULT_TASKS=('ocrbench_v2' 'mmbench_en_dev' 'mmmu_val' 'mathvista_testmini')
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

source uv_lmms_eval/bin/activate
cd src/lmms-eval

# for iteration in {1..10}; do
# echo "========================================="
# echo "Iteration $iteration of 10"
# echo "========================================="

for task in "${TASKS_ARR[@]}"; do
    echo "Evaluating task: $task"
    echo "Output path: ${OUTPUT_ROOT}/${MODEL}/${task}"
    LIMIT_ARGS=""
    if echo "$MODEL" | grep -qi "onevision"; then
        python3 -m lmms_eval \
                --model llava_onevision1_5 \
                --model_args pretrained="$MODEL" \
                --tasks "${task}" \
                --batch_size 1 \
                --log_samples \
                --limit 1000 \
                --log_samples_suffix "${task}" \
                --output_path "${OUTPUT_ROOT}/${MODEL}/${task}"
                #  --gen_kwargs "temperature=0,do_sample=False,top_p=1,seed=42" \
                # --overwrite
        # accelerate launch --num_processes=${NUM_GPUS} --main_process_port 12399 -m lmms_eval \
        #     --model llava_onevision1_5 \
        #     --model_args pretrained="$MODEL",attn_implementation=flash_attention_2,max_pixels=3240000 \
        #     --tasks "${task}" \
        #     --batch_size 512
    else
        python3 -m lmms_eval \
            --model vllm \
            --model_args model="${MODEL}",tensor_parallel_size="${NUM_GPUS}" \
            --tasks "${task}" \
            ${LIMIT_ARGS} \
            --batch_size 8192 \
            --log_samples \
            --limit 1000 \
            --log_samples_suffix "${task}" \
            --output_path "${OUTPUT_ROOT}/${MODEL}/${task}" \
            --gen_kwargs "temperature=0,do_sample=False,top_p=1,seed=42"
            # --overwrite
    fi
    sleep 20
done
# echo "Completed iteration $iteration of 10"
# echo ""
    # sleep 120
# done


# export TASKS="ocrbench_v2,mmbench_en_dev,mmmu_val,mathvista_testmini"
# MODEL=/path/to/Qwen2.5-VL-7B-Instruct
# OUTPUT_ROOT=outputs/eval_lmms
# ts -G 1 bash scripts/eval/lmms_eval/eval_lmms.sh "$MODEL" "$OUTPUT_ROOT"
