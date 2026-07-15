<h1 align="center">VFA: Empowering Multilingual MLLMs via Vision-Free Adaptation</h1>

<p align="center">
  <a href="https://openreview.net/forum?id=0stlElQEzm"><img src="https://img.shields.io/badge/Paper-OpenReview-red" alt="Paper"></a>
  <a href="https://huggingface.co/datasets/agentlans/multilingual-sft"><img src="https://img.shields.io/badge/Data-HuggingFace-yellow" alt="Data"></a>
  <a href="#"><img src="https://img.shields.io/badge/License-Apache_2.0-green.svg" alt="License"></a>
</p>

<p align="center">
  <b>Yixia Li<sup>1*&loz;</sup>, Yaqing Shi<sup>3*</sup>, Zhiwen Ruan<sup>1</sup>, Dongdong Zhang<sup>2</sup>, Lingjie Jiang<sup>4</sup>,<br>
  Shaohan Huang<sup>2</sup>, Yun Chen<sup>3</sup>, Guanhua Chen<sup>1&dagger;</sup>, Furu Wei<sup>2</sup></b>
</p>
<p align="center">
  <sup>1</sup>Southern University of Science and Technology &nbsp; <sup>2</sup>Microsoft Research Asia<br>
  <sup>3</sup>Shanghai University of Finance and Economics &nbsp; <sup>4</sup>Peking University<br>
  <sup>*</sup>Equal contribution &nbsp; <sup>&dagger;</sup>Corresponding author &nbsp; <sup>&loz;</sup>Work done during internship at Microsoft Research Asia.<br>

</p>

---

## Overview

**VFA (Vision-Free Adaptation)** is an efficient framework for enhancing the multilingual capabilities of Multimodal Large Language Models (MLLMs) without relying on scarce multilingual image-text data.

VFA decouples multilingual learning from visual alignment through a two-stage vision-free strategy:
- **Stage 1**: Fine-tune the base LLM on multilingual text-only data to derive a multilingual task vector
- **Stage 2**: Merge the multilingual task vector into the vision-aligned MLLM, preserving visual grounding while injecting multilingual competence

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/sustech-nlp/VFA.git
cd VFA
```

### 2. Install LLaMA-Factory

LLaMA-Factory is included locally at `src/LLaMA-Factory/`. Install it into a virtual environment:
```bash
bash scripts/env_setup/uv_llamafactory.sh
```

### 3. Install evaluation frameworks

**lmms-eval** (for multimodal benchmarks):
```bash
bash scripts/env_setup/uv_lmms_eval.sh
```

**OpenCompass** (for text-only benchmarks):
```bash
bash scripts/env_setup/uv_opencompass.sh
```

## Quick Start

### Stage 1: Fine-tune Base LLM on Multilingual Text Data

Run training (example with Qwen2.5-7B):
```bash
bash scripts/train/run_train.sh configs/finetune_llm/Qwen2.5-7B/multilingual_sft_99999.yaml
```

### Stage 2: Merge Multilingual Task Vector into MLLM

```bash
source uv_llamafactory/bin/activate
```

**Task Arithmetic**:
```bash
python scripts/merge/vlm_model_merge.py \
    --vlm_model_type Qwen/Qwen2.5-VL-7B-Instruct \
    --llm_model_type Qwen/Qwen2.5-7B \
    --vlm_model_path /path/to/Qwen2.5-VL-7B-Instruct \
    --llm_model_path /path/to/finetuned-Qwen2.5-7B \
    --base_model_path /path/to/Qwen2.5-7B \
    --output_dir outputs/Qwen2.5-VL-7B-VFA \
    --mode task_arithmetic \
    --alpha 0.8
```

**Weighted Averaging**:
```bash
python scripts/merge/vlm_model_merge.py \
    --vlm_model_type Qwen/Qwen2.5-VL-7B-Instruct \
    --llm_model_type Qwen/Qwen2.5-7B \
    --vlm_model_path /path/to/Qwen2.5-VL-7B-Instruct \
    --llm_model_path /path/to/finetuned-Qwen2.5-7B \
    --output_dir outputs/Qwen2.5-VL-7B-VFA \
    --mode weighted_average \
    --alpha 0.8
```

**TIES-Merging**:
```bash
python scripts/merge/vlm_model_merge.py \
    --vlm_model_type HuggingFaceM4/Idefics3-8B-Llama3 \
    --llm_model_type meta-llama/Llama-3.1-8B \
    --vlm_model_path /path/to/Idefics3-8B-Llama3 \
    --llm_model_path /path/to/finetuned-Llama-3.1-8B-Instruct \
    --base_model_path /path/to/Llama-3.1-8B \
    --output_dir outputs/Idefics3-8B-VFA \
    --mode ties \
    --alpha 0.8 \
    --density 0.2
```

### Stage 3: Evaluate

**Multilingual multimodal evaluation** (lmms-eval):
```bash
export TASKS="maxm,xgqa,xmmmu,xm100,marvl,m3exam"
bash scripts/eval/eval_pangeabench.sh /path/to/merged-model outputs/eval_results
```

**Text-only multilingual evaluation** (OpenCompass):
```bash
export TASKS="tydiqa_gen,mmmlu_gen_d5017d,xnli_gen_973734,mhellaswag_gen_1a6b73,mifeval_gen_79f8fb,mlogiqa_gen_36c4f9,flores_gen_2697d7"
bash scripts/eval/eval_opencompass.sh /path/to/merged-model outputs/eval_results chat "$TASKS"
```

**General multimodal evaluation** (lmms-eval):
```bash
export TASKS="ocrbench_v2,mmbench_en_dev,mmmu_val,mathvista_testmini"
bash scripts/eval/eval_lmms.sh /path/to/merged-model outputs/eval_results
```

## Released Checkpoints

The following trained checkpoints are available on ModelScope.

| Model | Training Type | Download |
|:------|:--------------|:---------|
| llama3-8b-instruct | Full SFT | [ModelScope](https://www.modelscope.cn/models/YQ0509/MMLM-via-Model-Merge/files?fileName=llama3-8b-instruct-full-sft-multilingual-sft-99999-epoch1-bs128) |
| llama31-8b-instruct | Full SFT | [ModelScope](https://www.modelscope.cn/models/YQ0509/MMLM-via-Model-Merge/files?fileName=llama31-8b-instruct-full-sft-multilingual_sft_99999-epoch1-bs128) |
| qwen25-7b-base | Full SFT | [ModelScope](https://www.modelscope.cn/models/YQ0509/MMLM-via-Model-Merge/files?fileName=qwen25-7b-base-full-sft-multilingual_sft_99999-epoch1-bs128) |
| qwen25-7b-instruct | Full SFT | [ModelScope](https://www.modelscope.cn/models/YQ0509/MMLM-via-Model-Merge/files?fileName=qwen25-7b-instruct-full-sft-multilingual_sft_99999-epoch1-bs128) |
| qwen3-4b-base | Full SFT | [ModelScope](https://www.modelscope.cn/models/YQ0509/MMLM-via-Model-Merge/files?fileName=qwen3-4b-base-full-sft-multilingual_sft_99999-epoch1-bs128) |
| qwen3-8b-base | Full SFT | [ModelScope](https://www.modelscope.cn/models/YQ0509/MMLM-via-Model-Merge/files?fileName=qwen3-8b-base-full-sft-multilingual_sft_99999-epoch1-bs128) |

## Supported Models

### VLM-LLM Pairs

| MLLM (VLM) | LLM Backbone | Backbone Family |
|:------------|:-------------|:----------------|
| Qwen2.5-VL-7B-Instruct | Qwen2.5-7B | Qwen |
| Idefics3-8B-Llama3 | Llama-3.1-8B / 8B-Instruct | Llama |
| llama3-llava-next-8b-hf | Meta-Llama-3-8B / 8B-Instruct | Llama |
| LLaVA-OneVision-1.5-8B-Instruct | Qwen3-8B-Base | Qwen |
| LLaVA-OneVision-1.5-4B-Instruct | Qwen3-4B-Base | Qwen |
| Qwen2-VL-7B-Instruct | Qwen2-7B | Qwen |
| Qwen3-VL-8B-Instruct | Qwen3-8B-Base | Qwen |

### Merging Methods

| Method | Formula | Description |
|:-------|:--------|:------------|
| Weighted Averaging | `merged = alpha * VLM + (1-alpha) * LLM` | Linear interpolation between VLM and fine-tuned LLM |
| Task Arithmetic | `merged = base + alpha * (TV_vlm + TV_ft)` | Adds weighted task vectors to the base model |
| TIES-Merging | `merged = base + alpha * TIES(TV_vlm, TV_ft)` | Trim, Elect sign, and disjoint merge to reduce interference |


## Data

We fine-tune base LLMs on a **100K-example subset** of the [Multilingual-SFT dataset](https://huggingface.co/datasets/agentlans/multilingual-sft).

To download and preprocess the dataset (filters out multimodal samples):
```bash
python scripts/data_processing/process_multilingual_sft.py
# Output: data/multilingual_sft_99999.json
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{li2026vfa,
    title={VFA: Empowering Multilingual MLLMs via Vision-Free Adaptation},
    author={Yixia Li and Yaqing Shi and Zhiwen Ruan and Dongdong Zhang and Lingjie Jiang and Shaohan Huang and Yun Chen and Guanhua Chen and Furu Wei},
    booktitle={Proceedings of ACL},
    year={2026}
}
```

## Acknowledgments

This project builds upon the following excellent open-source projects:
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for training infrastructure
- [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) and [Pangea](https://github.com/neulab/Pangea) for multimodal evaluation
- [OpenCompass](https://github.com/open-compass/opencompass) for text evaluation
