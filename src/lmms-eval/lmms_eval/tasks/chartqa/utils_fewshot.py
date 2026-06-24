"""Utilities for chartqa_fewshot task (HuggingFaceM4/ChartQA dataset).

Field mapping from HuggingFaceM4/ChartQA → lmms-lab/ChartQA:
  query            → question
  label (list[str]) → answer (str)
  human_or_machine (ClassLabel: 0=human,1=machine) → type (human_test/augmented_test)
"""


def chartqa_fewshot_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def chartqa_fewshot_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["query"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


def chartqa_fewshot_process_results(doc, results):
    pred = results[0]
    # human_or_machine: 0=human, 1=machine
    is_human = doc["human_or_machine"] == 0
    # label is a list, take the first element as the gold answer
    gold_answer = doc["label"][0] if isinstance(doc["label"], list) else doc["label"]
    score = relaxed_correctness(pred, gold_answer)
    score = 1.0 if score else 0.0
    return_dict = {"relaxed_overall": score}
    if is_human:
        return_dict["relaxed_human_split"] = score
    else:
        return_dict["relaxed_augmented_split"] = score
    return return_dict


def chartqa_fewshot_doc_to_target(doc):
    """Return the gold answer as a string for few-shot labeled examples."""
    if isinstance(doc["label"], list):
        return doc["label"][0]
    return doc["label"]


def relaxed_correctness(prediction, target, max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    "Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct."

    This function is taken from https://github.com/QwenLM/Qwen-VL/blob/34b4c0ee7b07726371b960911f249fe61b362ca3/eval_mm/evaluate_vqa.py#L113
    """

    def _to_float(text: str):
        try:
            if text.endswith("%"):
                return float(text.rstrip("%")) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()
