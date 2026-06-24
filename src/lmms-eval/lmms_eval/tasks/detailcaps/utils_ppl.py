"""Utility helpers for the DetailCaps PPL evaluation task."""

import io
import math

from PIL import Image


def detailcaps_doc_to_visual(doc: dict):
    """Return the image as a list."""
    return [Image.open(io.BytesIO(doc["binary"])).convert("RGB")]


def detailcaps_doc_to_target_ppl(doc: dict) -> str:
    """Return the GPT-4o caption for PPL computation."""
    return doc["GT_Caption_GPT4O"]


def detailcaps_ppl_process_results(doc: dict, results) -> dict:
    """Store per-sample cross-entropy loss for PPL aggregation."""
    loss, is_greedy = results[0]
    return {"detailcaps_ppl": float(loss)}


def detailcaps_ppl_aggregate(items, **kwargs) -> float:
    """PPL = exp(mean cross-entropy loss)."""
    return math.exp(sum(items) / len(items))
