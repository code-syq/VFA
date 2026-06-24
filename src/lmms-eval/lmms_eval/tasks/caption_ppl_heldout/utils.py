"""Utility helpers for the held-out caption PPL evaluation task.

This task computes caption perplexity on a held-out subset of the actual
training data (LLaVA-OneVision-1.5-Mid-Training-Webdataset), ensuring the
PPL metric faithfully tracks training progress without distribution mismatch.
"""

import math


def heldout_doc_to_visual(doc: dict):
    """Return the image as a list (already decoded by HF Dataset Image feature)."""
    image = doc["image"]
    if hasattr(image, "convert"):
        return [image.convert("RGB")]
    return [image]


def heldout_doc_to_target(doc: dict) -> str:
    """Return the caption for PPL computation."""
    return doc["caption"]


def heldout_ppl_process_results(doc: dict, results) -> dict:
    """Store per-sample cross-entropy loss for PPL aggregation."""
    loss, is_greedy = results[0]
    return {"caption_ppl": float(loss)}


def heldout_ppl_aggregate(items, **kwargs) -> float:
    """PPL = exp(mean cross-entropy loss)."""
    return math.exp(sum(items) / len(items))
