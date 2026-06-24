"""Prism-mode utilities for cvqa.

Replaces image input with pre-computed GPT-5 captions.
"""

import json
import os
from pathlib import Path
from functools import lru_cache

# Import original utils for metrics/processing
from lmms_eval.tasks.cvqa import utils as original_utils


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


def prism_doc_to_text(doc, model_specific_prompt_kwargs=None):
    """Inject caption into text prompt for cvqa."""
    captions = _load_captions("cvqa", "test")

    doc_id = f"idx_{doc.get('_idx', '')}"
    caption_data = captions.get(doc_id, {})
    caption = caption_data.get("query_specific_caption", caption_data.get("generic_caption", "[NO_CAPTION]"))

    # Use original text construction
    text = original_utils.cvqa_doc_to_text(doc, model_specific_prompt_kwargs)

    return f"Image description: {caption}\n\n{text}"


cvqa_doc_to_target = original_utils.cvqa_doc_to_target
cvqa_process_results = original_utils.cvqa_process_results
cvqa_test_aggregation_result = original_utils.cvqa_test_aggregation_result
