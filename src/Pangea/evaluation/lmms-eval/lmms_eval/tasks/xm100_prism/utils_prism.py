"""Prism-mode utilities for xm100.

Replaces image input with pre-computed GPT-5 captions.
"""

import json
import os
from pathlib import Path
from functools import lru_cache

# Import original utils for metrics/processing
from lmms_eval.tasks.xm100 import utils as original_utils


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
    """For xm100 (captioning task), just describe the image."""
    split = doc.get("_split", "en")
    captions = _load_captions("xm100", split)

    doc_id = f"idx_{doc.get('_idx', '')}"
    caption_data = captions.get(doc_id, {})
    caption = caption_data.get("generic_caption", "[NO_CAPTION]")

    # xm100 is a captioning task — the caption IS the answer in Prism mode
    lang = doc.get("language", split)
    return f"Based on the following image description, generate a caption in {lang}.\n\nImage description: {caption}\n\nCaption:"


# Re-export
from lmms_eval.tasks.xm100.utils import *
