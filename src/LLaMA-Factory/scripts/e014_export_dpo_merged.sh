#!/usr/bin/env bash
set -euo pipefail

cd /nfsdata/syq/LLaMA-Factory/src/LLaMA-Factory
mkdir -p /nfsdata/syq/logs /nfsdata/syq/models/dpo_merged

/home/syq/miniconda3/envs/llamafactory/bin/llamafactory-cli export \
  examples/merge_lora/e014_rlaif_v_dpo/qwen25vl_original_rlaifv10k_lora_dpo_export.yaml \
  > /nfsdata/syq/logs/e014_export_original_dpo_merged_gpu6.log 2>&1

/home/syq/miniconda3/envs/llamafactory/bin/llamafactory-cli export \
  examples/merge_lora/e014_rlaif_v_dpo/qwen25vl_layers12_27_wa0p9_rlaifv10k_lora_dpo_export.yaml \
  > /nfsdata/syq/logs/e014_export_layers12_27_wa0p9_dpo_merged_gpu6.log 2>&1
