import os
import json
import argparse
import re
from typing import Callable, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModel, AutoProcessor

try:
    from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
except ImportError:
    LlavaNextForConditionalGeneration = AutoModelForCausalLM
    LlavaNextProcessor = AutoProcessor

try:
    from transformers import Qwen2VLForConditionalGeneration
except ImportError:
    Qwen2VLForConditionalGeneration = AutoModelForCausalLM

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    Qwen2_5_VLForConditionalGeneration = AutoModelForCausalLM

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = AutoModelForCausalLM

try:
    from transformers import Idefics3ForConditionalGeneration
except ImportError:
    Idefics3ForConditionalGeneration = AutoModelForCausalLM

try:
    from transformers import Idefics2ForConditionalGeneration
except ImportError:
    Idefics2ForConditionalGeneration = AutoModelForCausalLM

import merge_utils


VLM_MODEL_NAMES = {
    "HuggingFaceM4/idefics2-8b": {
        "model_class": Idefics2ForConditionalGeneration,
        "processor_class": AutoProcessor,
        "language_model_key": "model.text_model",
    },
    "llava-hf/llava-v1.6-mistral-7b-hf": {
        "model_class": LlavaNextForConditionalGeneration,
        "processor_class": LlavaNextProcessor,
    },
    "llava-hf/llama3-llava-next-8b-hf": {
        "model_class": LlavaNextForConditionalGeneration,
        "processor_class": LlavaNextProcessor,
    },
    "HuggingFaceM4/Idefics3-8B-Llama3": {
        "model_class": Idefics3ForConditionalGeneration,
        "processor_class": AutoProcessor,
    },
    "OpenGVLab/InternVL3-8B-hf": {
        "model_class": AutoModel,
        "processor_class": AutoProcessor,
    },
    "Qwen/Qwen2-VL-7B-Instruct": {
        "model_class": Qwen2VLForConditionalGeneration,
        "processor_class": AutoProcessor,
    },
    "Qwen/Qwen2.5-VL-7B-Instruct": {
        "model_class": Qwen2_5_VLForConditionalGeneration,
        "processor_class": AutoProcessor,
    },
    "Qwen/Qwen3-VL-4B-Instruct": {
        "model_class": Qwen3VLForConditionalGeneration,
        "processor_class": AutoProcessor,
    },
    "Qwen/Qwen3-VL-8B-Instruct": {
        "model_class": Qwen3VLForConditionalGeneration,
        "processor_class": AutoProcessor,
    },
    "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct": {
        "model_class": AutoModelForCausalLM,
        "processor_class": AutoProcessor,
    },
    "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct": {
        "model_class": AutoModelForCausalLM,
        "processor_class": AutoProcessor,
    },
}

LLM_MODEL_NAMES = {
    "mistralai/Mistral-7B-v0.1": {
        "model_class": AutoModelForCausalLM,
    },
    "mistralai/Mistral-7B-Instruct-v0.2": {
        "model_class": AutoModelForCausalLM,
    },
    "meta-llama/Llama-3.1-8B": {
        "model_class": AutoModelForCausalLM,
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "model_class": AutoModelForCausalLM,
    },
    "meta-llama/Meta-Llama-3-8B": {
        "model_class": AutoModelForCausalLM,
    },
    "meta-llama/Meta-Llama-3-8B-Instruct": {
        "model_class": AutoModelForCausalLM,
    },
    "Qwen/Qwen2-7B": {
        "model_class": AutoModelForCausalLM,
    },
    "Qwen/Qwen2.5-7B": {
        "model_class": AutoModelForCausalLM,
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "model_class": AutoModelForCausalLM,
    },
    "Qwen/Qwen3-4B-Base": {
        "model_class": AutoModelForCausalLM,
    },
    "Qwen/Qwen3-4B-Instruct": {
        "model_class": AutoModelForCausalLM,
    },
    "Qwen/Qwen3-8B-Base": {
        "model_class": AutoModelForCausalLM,
    },
}

LLM_VLM_MAPPINGS = {
    "mistralai/Mistral-7B-v0.1": {
        "HuggingFaceM4/idefics2-8b": lambda llm_key: llm_key.replace("model", "model.text_model"),
    },
    "mistralai/Mistral-7B-Instruct-v0.2": {
        "llava-hf/llava-v1.6-mistral-7b-hf": lambda llm_key: llm_key.replace("model", "model.language_model"),
    },
    "meta-llama/Meta-Llama-3-8B": {
        "llava-hf/llama3-llava-next-8b-hf": lambda llm_key: llm_key.replace("model", "model.language_model"),
    },
    "meta-llama/Meta-Llama-3-8B-Instruct": {
        "llava-hf/llama3-llava-next-8b-hf": lambda llm_key: llm_key.replace("model", "model.language_model"),
    },
    "meta-llama/Llama-3.1-8B": {
        "HuggingFaceM4/Idefics3-8B-Llama3": lambda llm_key: llm_key.replace("model", "model.text_model"),
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "HuggingFaceM4/Idefics3-8B-Llama3": lambda llm_key: llm_key.replace("model", "model.text_model"),
    },
    "Qwen/Qwen2-7B": {
        "Qwen/Qwen2-VL-7B-Instruct": lambda llm_key: llm_key.replace("model", "model.language_model"),
    },
    "Qwen/Qwen2.5-7B": {
        "Qwen/Qwen2.5-VL-7B-Instruct": lambda llm_key: llm_key.replace("model", "model.language_model"),
        "OpenGVLab/InternVL3-8B-hf": lambda llm_key: llm_key.replace("model", "language_model"),
    },
    "Qwen/Qwen3-8B-Base": {
        "Qwen/Qwen3-VL-8B-Instruct": lambda llm_key: llm_key.replace("model", "model.language_model"),
        "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct": lambda llm_key: llm_key.replace("model", "model.language_model"),
    },
    "Qwen/Qwen3-4B-Base": {
        "Qwen/Qwen3-VL-4B-Instruct": lambda llm_key: llm_key.replace("model", "model.language_model"),
        "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct": lambda llm_key: llm_key.replace("model", "model.language_model"),
    },
}

def valid_model_pairs():
    paris = []
    for LLM in LLM_VLM_MAPPINGS.keys():
        for VLM in LLM_VLM_MAPPINGS[LLM].keys():
            paris.append((LLM, VLM))
    return paris

def verify_mapping(MODELS_ROOT):
    for LLM in LLM_VLM_MAPPINGS.keys():
        LLM_model = LLM_MODEL_NAMES[LLM]["model_class"].from_pretrained(os.path.join(MODELS_ROOT, LLM), trust_remote_code=True)
        LLM_state_dict = LLM_model.state_dict()
        for VLM in LLM_VLM_MAPPINGS[LLM].keys():
            print("\n\n#####", LLM, VLM, "#####")
            VLM_model = VLM_MODEL_NAMES[VLM]["model_class"].from_pretrained(os.path.join(MODELS_ROOT, VLM), trust_remote_code=True)
            MAPPING = LLM_VLM_MAPPINGS[LLM][VLM]
            VLM_state_dict = VLM_model.state_dict()
            for key in LLM_state_dict.keys():
                if MAPPING(key) not in VLM_state_dict.keys():
                    print(LLM, VLM, key, "not found")

def _is_excluded_key(key: str) -> bool:
    if "embed_tokens" in key or "lm_head" in key:
        return True
    return False


def _extract_layer_idx(key: str) -> Optional[int]:
    # Match common patterns like "...layers.12..." / "...h.12..."
    m = re.search(r"\.(\d+)\.", key)
    if not m:
        return None
    return int(m.group(1))


def _parse_layer_range(layer_range: Optional[str]) -> Optional[tuple[int, int]]:
    if layer_range is None:
        return None
    m = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", layer_range)
    if not m:
        raise ValueError(f"Invalid --layer_range: {layer_range}. Expected format: start-end (e.g., 0-15)")
    start, end = int(m.group(1)), int(m.group(2))
    if start > end:
        raise ValueError(f"Invalid --layer_range: {layer_range}. start must be <= end.")
    return (start, end)


def _orthogonalize(target: torch.Tensor, reference: torch.Tensor, eps: float) -> torch.Tensor:
    target_f = target.to(torch.float32)
    reference_f = reference.to(torch.float32)
    denom = torch.sum(reference_f * reference_f).clamp_min(eps)
    coef = torch.sum(target_f * reference_f) / denom
    return target - (coef.to(reference.dtype) * reference)

def _load_vlm_and_processor(vlm_model_type: str, vlm_model_path: str):
    print(f"Loading VLM model from {vlm_model_path}")
    info = VLM_MODEL_NAMES[vlm_model_type]
    model = info["model_class"].from_pretrained(vlm_model_path, trust_remote_code=True)
    processor = info["processor_class"].from_pretrained(vlm_model_path, trust_remote_code=True)
    return model, processor

def _load_llm(llm_model_type: str, llm_model_path: str):
    print(f"Loading LLM model from {llm_model_path}")
    info = LLM_MODEL_NAMES[llm_model_type]
    return info["model_class"].from_pretrained(llm_model_path, trust_remote_code=True)


@torch.no_grad()
def merge_vlm_llm(
    vlm_model_type: str,
    llm_model_type: str,
    vlm_model_path: str,
    llm_model_path: str,
    output_dir: str,
    alpha: float,
    alpha2: Optional[float] = None,
    mode: str = "weighted_average",
    base_model_path: Optional[str] = None,
    density: float = 0.2,
    include_regex: Optional[str] = None,
    exclude_regex: Optional[str] = None,
    layer_range: Optional[str] = None,
    alpha2_regex: Optional[str] = None,
    orth_eps: float = 1e-12,
    show_progress: bool = True,
) -> None:
    assert mode in ["base", "weighted_average", "task_arithmetic", "orth_task_arithmetic", "ties", "dareties", "darelinear"]
    effective_mode = "weighted_average" if mode == "base" else mode
    include_pat = re.compile(include_regex) if include_regex else None
    exclude_pat = re.compile(exclude_regex) if exclude_regex else None
    alpha2_pat = re.compile(alpha2_regex) if alpha2_regex else None
    layer_bounds = _parse_layer_range(layer_range)

    merge_config = {
        "vlm_model_type": vlm_model_type,
        "llm_model_type": llm_model_type,
        "vlm_model_path": vlm_model_path,
        "llm_model_path": llm_model_path,
        "base_model_path": base_model_path,
        "mode": mode,
        "effective_mode": effective_mode,
        "alpha": alpha,
        "alpha2": alpha2,
        "density": density,
        "include_regex": include_regex,
        "exclude_regex": exclude_regex,
        "layer_range": layer_range,
        "alpha2_regex": alpha2_regex,
        "orth_eps": orth_eps,
    }
    print("### Merge config ###")
    print(json.dumps(merge_config, ensure_ascii=False, indent=4))

    if llm_model_type not in LLM_VLM_MAPPINGS or vlm_model_type not in LLM_VLM_MAPPINGS[llm_model_type]:
        raise ValueError(
            f"Missing mapping for llm_model_type={llm_model_type} -> vlm_model_type={vlm_model_type}. "
            f"Please add to LLM_VLM_MAPPINGS."
        )
    mapping_fn: Callable[[str], str] = LLM_VLM_MAPPINGS[llm_model_type][vlm_model_type]

    vlm_model, processor = _load_vlm_and_processor(vlm_model_type, vlm_model_path)
    llm_model = _load_llm(llm_model_type, llm_model_path)

    vlm_state_dict = vlm_model.state_dict()
    llm_state_dict = llm_model.state_dict()

    base_state_dict = None
    if effective_mode in {"task_arithmetic", "orth_task_arithmetic", "ties", "dareties", "darelinear"}:
        if not base_model_path:
            raise ValueError(f"mode={mode} requires --base_model_path")
        base_model = _load_llm(llm_model_type, base_model_path)
        base_state_dict = base_model.state_dict()

    keys = list(llm_state_dict.keys())
    it = tqdm(keys, desc=f"merge({mode})", total=len(keys)) if show_progress else keys

    skipped_keys = []
    for llm_key in it:
        if _is_excluded_key(llm_key):
            skipped_keys.append(llm_key)
            continue
        if include_pat and include_pat.search(llm_key) is None:
            skipped_keys.append(llm_key)
            continue
        if exclude_pat and exclude_pat.search(llm_key):
            skipped_keys.append(llm_key)
            continue
        if layer_bounds is not None:
            layer_idx = _extract_layer_idx(llm_key)
            if layer_idx is None or not (layer_bounds[0] <= layer_idx <= layer_bounds[1]):
                skipped_keys.append(llm_key)
                continue

        mapped_key = mapping_fn(llm_key)
        if mapped_key not in vlm_state_dict:
            raise KeyError(
                f"Mapped key not found in VLM state_dict.\n"
                f"  llm_key={llm_key}\n"
                f"  mapped_key={mapped_key}\n"
                f"  vlm_model_type={vlm_model_type}\n"
                f"  llm_model_type={llm_model_type}\n"
                f"Hint: run verify_mapping() and/or fix LLM_VLM_MAPPINGS."
            )
        vlm_w = vlm_state_dict[mapped_key]
        llm_w = llm_state_dict[llm_key].to(device=vlm_w.device, dtype=vlm_w.dtype)
        use_alpha2 = alpha2 is not None and alpha2_pat is not None and alpha2_pat.search(llm_key) is not None
        alpha_for_key = float(alpha2) if use_alpha2 else float(alpha)

        if effective_mode == "weighted_average":
            # vlm = alpha*vlm + (1-alpha)*llm
            vlm_w.mul_(alpha_for_key).add_(llm_w, alpha=(1.0 - alpha_for_key))
            continue

        if effective_mode in {"task_arithmetic", "orth_task_arithmetic", "ties", "dareties", "darelinear"}:
            assert base_state_dict is not None
            if llm_key not in base_state_dict:
                raise ValueError(f"Key {llm_key} not found in base model")

            base_w = base_state_dict[llm_key].to(device=vlm_w.device, dtype=vlm_w.dtype)
            if base_w.shape != vlm_w.shape or base_w.shape != llm_w.shape:
                raise ValueError(f"Shape mismatch: {base_w.shape} != {vlm_w.shape} != {llm_w.shape}")

            tv_vlm = vlm_w - base_w
            tv_llm = llm_w - base_w
            weights = torch.tensor([float(alpha), alpha_for_key], dtype=torch.float32, device=tv_vlm.device)
            if effective_mode == "task_arithmetic":
                mix = merge_utils.task_arithmetic([tv_vlm, tv_llm], weights)
            elif effective_mode == "orth_task_arithmetic":
                tv_llm_orth = _orthogonalize(tv_llm, tv_vlm, eps=orth_eps)
                mix = merge_utils.task_arithmetic([tv_vlm, tv_llm_orth], weights)
            elif effective_mode == "ties":
                mix = merge_utils.ties([tv_vlm, tv_llm], weights, density)
            elif effective_mode == "dareties":
                mix = merge_utils.dare_ties([tv_vlm, tv_llm], weights, density)
            else:  # darelinear
                mix = merge_utils.dare_linear([tv_vlm, tv_llm], weights, density)
            vlm_w.copy_(base_w + mix.to(dtype=vlm_w.dtype))
            continue

        raise ValueError(f"Unknown mode: {mode}")

    os.makedirs(output_dir, exist_ok=True)
    vlm_model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    merge_config["skipped_keys"] = skipped_keys
    with open(os.path.join(output_dir, "merge_config.json"), "w", encoding="utf-8") as f:
        json.dump(merge_config, f, ensure_ascii=False, indent=4)

    print(f"[OK] Saved merged VLM to: {output_dir}")
    print(f"[INFO] skipped_keys: {skipped_keys}")


def main():
    parser = argparse.ArgumentParser(description="Merge an LLM into a VLM (local load + key mapping) and save as HF model dir.")
    parser.add_argument("--vlm_model_type", type=str, required=True, help="Only used to identify VLM type for registry/mapping (xx/xx).")
    parser.add_argument("--llm_model_type", type=str, required=True, help="Only used to identify LLM type for registry/mapping (xx/xx).")
    parser.add_argument("--vlm_model_path", type=str, required=True, help="Path to load the VLM from.")
    parser.add_argument("--llm_model_path", type=str, required=True, help="Path to load the LLM from.")
    parser.add_argument("--base_model_path", type=str, default=None, help="Path to load the base model from (ties/dare*). Model class is decided by llm_model_type.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--mode", type=str, default="weighted_average", choices=["base", "weighted_average", "task_arithmetic", "orth_task_arithmetic", "ties", "dareties", "darelinear"])
    parser.add_argument("--alpha", type=float, required=True, help="For weighted_average mode, set alpha for VLM and (1-alpha) for LLM. For others, set alpha for both VLM and LLM.")
    parser.add_argument("--alpha2", type=float, default=None, help="Optional secondary alpha. Only applied to keys matched by --alpha2_regex.")
    parser.add_argument("--density", type=float, default=0.2, help="Density for ties/dare* modes.")
    parser.add_argument("--include_regex", type=str, default=None, help="Only merge keys that match this regex.")
    parser.add_argument("--exclude_regex", type=str, default=None, help="Skip keys that match this regex.")
    parser.add_argument("--layer_range", type=str, default=None, help="Only merge keys in [start-end], e.g., 0-15.")
    parser.add_argument("--alpha2_regex", type=str, default=None, help="Regex for keys that should use --alpha2.")
    parser.add_argument("--orth_eps", type=float, default=1e-12, help="Numerical epsilon used in orthogonalized merge.")
    parser.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bar.")

    args = parser.parse_args()

    if (args.llm_model_type, args.vlm_model_type) not in valid_model_pairs():
        print("Valid model pairs:")
        for pair in valid_model_pairs():
            print(pair)
        raise ValueError(f"Invalid model pair: {args.vlm_model_type} {args.llm_model_type}")

    if args.mode in {"task_arithmetic", "orth_task_arithmetic", "ties", "dareties", "darelinear"} and args.base_model_path is None:
        raise ValueError(f"mode={args.mode} requires --base_model_path")

    vlm_model_path = args.vlm_model_path
    llm_model_path = args.llm_model_path

    merge_vlm_llm(
        vlm_model_type=args.vlm_model_type,
        llm_model_type=args.llm_model_type,
        vlm_model_path=vlm_model_path,
        llm_model_path=llm_model_path,
        output_dir=args.output_dir,
        alpha=args.alpha,
        alpha2=args.alpha2,
        mode=args.mode,
        base_model_path=args.base_model_path,
        density=args.density,
        include_regex=args.include_regex,
        exclude_regex=args.exclude_regex,
        layer_range=args.layer_range,
        alpha2_regex=args.alpha2_regex,
        orth_eps=args.orth_eps,
        show_progress=(not args.no_progress),
    )


if __name__ == "__main__":
    main()
    # verify_mapping(MODELS_ROOT="/path/to/models")



"""
## base mode
# only need to specify model type
VLM_MODEL_TYPE=llava-hf/llama3-llava-next-8b-hf
LLM_MODEL_TYPE=meta-llama/Meta-Llama-3-8B-Instruct


# path to models
VLM_MODEL_PATH=llava-hf/llama3-llava-next-8b-hf
LLM_MODEL_PATH=meta-llama/Meta-Llama-3-8B-Instruct

# output directory
OUTPUT_DIR=/nfsdata/syq/models/merge-model/debug_weighted_average/llama3-llava-next-8b-hf-merged-with-Llama-3-8B-WA-0.9

# merge args
MODE=weighted_average
ALPHA=0.9

python vlm_model_merge.py \
    --vlm_model_type $VLM_MODEL_TYPE \
    --llm_model_type $LLM_MODEL_TYPE \
    --vlm_model_path $VLM_MODEL_PATH \
    --llm_model_path $LLM_MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --mode $MODE \
    --alpha $ALPHA
"""

"""
## task arithmetic mode
# only need to specify model type
VLM_MODEL_TYPE=Qwen/Qwen2.5-VL-7B-Instruct
LLM_MODEL_TYPE=Qwen/Qwen3-4B-Base

# path to models
VLM_MODEL_PATH=/path/to/Qwen3-VL-4B-Instruct
LLM_MODEL_PATH=/path/to/Qwen3-4B-Instruct-2507
BASE_MODEL_PATH=/path/to/Qwen3-4B-Base

# output directory
OUTPUT_DIR=./debug_task_arithmetic

# merge args
MODE=task_arithmetic
ALPHA=0.7

ts python scripts/model_merge/vlm_model_merge.py \
    --vlm_model_type $VLM_MODEL_TYPE \
    --llm_model_type $LLM_MODEL_TYPE \
    --vlm_model_path $VLM_MODEL_PATH \
    --llm_model_path $LLM_MODEL_PATH \
    --base_model_path $BASE_MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --mode $MODE \
    --alpha $ALPHA
"""



"""
## ties mode
# only need to specify model type
VLM_MODEL_TYPE=Qwen/Qwen3-VL-4B-Instruct
LLM_MODEL_TYPE=Qwen/Qwen3-4B-Base

# path to models
VLM_MODEL_PATH=/path/to/Qwen3-VL-4B-Instruct
LLM_MODEL_PATH=/path/to/Qwen3-4B-Instruct-2507
BASE_MODEL_PATH=/path/to/Qwen3-4B-Base

# output directory
OUTPUT_DIR=./debug_ties

# merge args
MODE=ties
ALPHA=0.7
DENSITY=0.2

ts python scripts/model_merge/vlm_model_merge.py \
    --vlm_model_type $VLM_MODEL_TYPE \
    --llm_model_type $LLM_MODEL_TYPE \
    --vlm_model_path $VLM_MODEL_PATH \
    --llm_model_path $LLM_MODEL_PATH \
    --base_model_path $BASE_MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --mode $MODE \
    --alpha $ALPHA \
    --density $DENSITY
"""
