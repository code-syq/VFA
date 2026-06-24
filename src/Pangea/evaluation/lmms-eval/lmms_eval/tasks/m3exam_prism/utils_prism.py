"""Prism-mode utilities for m3exam.

Replaces image input with pre-computed GPT-5 captions.
"""

import json
import os
from pathlib import Path
from functools import lru_cache

# Import original utils for metrics/processing
from lmms_eval.tasks.m3exam import utils as original_utils


# Cache loaded captions
_caption_cache = {}


def _load_captions(task_name: str, split: str) -> dict:
    """Load captions from JSONL file."""
    cache_key = f"{task_name}_{split}"
    if cache_key in _caption_cache:
        return _caption_cache[cache_key]

    caption_dir = os.environ.get("CAPTION_DIR", "captions")
    caption_file = Path(caption_dir) / f"{task_name}_{split}.jsonl"

    captions = {}
    if caption_file.exists():
        with open(caption_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    captions[str(data["doc_id"])] = data
                    # Also index by doc_idx for datasets without explicit IDs
                    if "doc_idx" in data:
                        captions[f"idx_{data['doc_idx']}"] = data
                except (json.JSONDecodeError, KeyError):
                    continue

    _caption_cache[cache_key] = captions
    return captions


def prism_doc_to_visual(doc):
    """Return None — no image needed in Prism mode."""
    return None


def prism_doc_to_text(doc):
    """Inject caption into text prompt for m3exam."""
    split = doc.get("_split", "english")
    captions = _load_captions("m3exam", split)

    doc_id = f"idx_{doc.get('_idx', '')}"
    caption_data = captions.get(doc_id, {})
    caption = caption_data.get("query_specific_caption", caption_data.get("generic_caption", ""))

    # Use original text construction
    text = original_utils.m3exam_doc_to_text(doc)

    if caption and caption != "[NO_IMAGE]":
        return f"Image description: {caption}\n\n{text}"
    return text


m3exam_doc_to_visual_original = original_utils.m3exam_doc_to_visual
m3exam_process_results = original_utils.m3exam_process_results
m3exam_aggregate_results = original_utils.m3exam_aggregate_results
