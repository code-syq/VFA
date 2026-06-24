
# pip3 install vllm
# pip3 install qwen_vl_utils

# cd ~/prod/lmms-eval-public
# pip3 install -e .
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=18000000
export NCCL_DEBUG=DEBUG
export VLLM_ENABLE_V1_MULTIPROCESSING=0

# LLM-as-judge config for MathVerse (defaults to Azure judge).
export API_TYPE=${API_TYPE:-azure}
export MODEL_VERSION=${MODEL_VERSION:-gpt-4o}
export AZURE_ENDPOINT=${AZURE_ENDPOINT:-${AZURE_OPENAI_API_BASE:-${AZURE_OPENAI_ENDPOINT:-https://conversationhubeastus.openai.azure.com/}}}
export AZURE_API_KEY=${AZURE_API_KEY:-${AZURE_OPENAI_API_KEY:-${OPENAI_API_KEY:-}}}
export AZURE_USE_AAD=${AZURE_USE_AAD:-1}
# lmms_eval currently reads API version with both names in different modules.
export API_VERSION=${API_VERSION:-2024-02-15-preview}
export AZURE_API_VERSION=${AZURE_API_VERSION:-${API_VERSION}}

if [[ "${API_TYPE}" == "azure" ]]; then
    : "${AZURE_ENDPOINT:?Please set AZURE_ENDPOINT or AZURE_OPENAI_API_BASE, e.g. https://<resource>.openai.azure.com/}"
    echo "Using Azure endpoint: ${AZURE_ENDPOINT}"
    aad_flag=$(echo "${AZURE_USE_AAD:-1}" | tr '[:upper:]' '[:lower:]')
    if [[ -z "${AZURE_API_KEY}" && ("${aad_flag}" == "1" || "${aad_flag}" == "true" || "${aad_flag}" == "yes") ]]; then
        token_cmd=(az account get-access-token --resource https://cognitiveservices.azure.com --query accessToken -o tsv)
        if [[ -n "${AZURE_TENANT_ID:-}" ]]; then
            token_cmd+=(--tenant "${AZURE_TENANT_ID}")
        fi
        AZURE_OPENAI_AD_TOKEN="$("${token_cmd[@]}" 2>/dev/null || true)"
        export AZURE_OPENAI_AD_TOKEN
    fi
    if [[ -z "${AZURE_API_KEY}" && -z "${AZURE_OPENAI_AD_TOKEN:-}" ]]; then
        echo "Azure auth is missing. Run 'az login' (and optionally set AZURE_TENANT_ID), or provide AZURE_OPENAI_AD_TOKEN." >&2
        exit 1
    fi
fi


# task=mathverse_testmini_text_only
# task=mathverse_testmini_text_lite,mathverse_testmini_text_dominant,mathverse_testmini_text_only

task=mathverse_testmini_text_lite,mathverse_testmini_text_dominant,mathverse_testmini_text_only,mathverse_testmini_vision_only,mathverse_testmini_vision_dominant,mathverse_testmini_vision_intensive
tp_size=${TP_SIZE:-1}

python3 -m lmms_eval \
    --model vllm \
    --model_args model=/mnt/yixiali/CODES/LLaMA-Factory/outputs/vlmevalkit/debug_weighted_average/merge_Idefics3-8B-Llama3_merged-with_Llama-3.1-8B-Base_WA_0.7,tensor_parallel_size=${tp_size} \
    --tasks ${task} \
    --batch_size 64 \
    --log_samples \
    --limit 200 \
    --write_out \
    --log_samples_suffix ${task} \
    --output_path ./logs \
    --gen_kwargs "temperature=0,do_sample=False,top_p=1,seed=42"


# debug_weighted_average
# /mnt/yixiali/MODELS/Qwen/Qwen2.5-VL-7B-Instruct
# /mnt/yixiali/CODES/LLaMA-Factory/outputs/vlmevalkit/debug_task_arithmetic/merge_Qwen2.5-VL-7B-Instruct_merged-with_Qwen2.5-7B-Base_TA_0.9
# /mnt/yixiali/CODES/LLaMA-Factory/outputs/vlmevalkit/debug_task_arithmetic/merge_Qwen2.5-VL-7B-Instruct_merged-with_Qwen2.5-7B-Instruct_TA_0.9
# /mnt/yixiali/MODELS/HuggingFaceM4/Idefics3-8B-Llama3
# /mnt/yixiali/MODELS/meta-llama/Llama-3.1-8B-Instruct
# /mnt/yixiali/MODELS/meta-llama/Meta-Llama-3-8B

