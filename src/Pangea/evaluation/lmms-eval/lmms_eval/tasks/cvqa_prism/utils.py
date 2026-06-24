import re
import random
from difflib import SequenceMatcher
import os

from lmms_eval.tasks._task_utils. file_utils import generate_submission_file

def cvqa_doc_to_text(doc, model_specific_prompt_kwargs):
    if model_specific_prompt_kwargs["translated"] is True:
        print("Using translated prompt")
        question, choices = doc["Translated Question"], doc["Translated Options"]
        doc["Question"] = doc["Translated Question"]
        doc["Options"] = doc["Translated Options"]
    else:
        question, choices = doc["Question"], doc["Options"]
    len_choices = len(choices)
    options = [chr(ord("A") + i) for i in range(len_choices)]
    choices_str = "\n".join([f"{option}.  {choice}" for option, choice in zip(options, choices)])
    return f"Question: {question} \n\n Options: {choices_str} \n\n Answer with the option's letter from the given choices directly."


def cvqa_doc_to_visual(doc):
    if doc["image"] is None:
        return []
    return [doc["image"]. convert("RGB")]


def cvqa_doc_to_target(doc):
    len_choices = len(doc["Options"])
    options = [chr(ord("A") + i) for i in range(len_choices)]
    return options[doc["Label"]]


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def parse_multi_choice_response(response, options):
    response = response.strip()

    # Original letter-matching logic
    match = re.search(r'\(?([A-D])[).:\s]', response, re. IGNORECASE)
    if match:
        return match.group(1).upper()

    # If no match found, fallback to searching for any A, B, C, or D in the response
    match = re.search(r'[ABCD]', response, re.IGNORECASE)
    if match:
        return match.group(0).upper()

    # If no letter found, match full content
    best_match = None
    best_match_ratio = 0
    for i, option in enumerate(options):
        option_content = re. sub(r'^[A-D]\.\s*', '', option).strip()
        similarity = similar(response, option_content)
        if similarity > best_match_ratio:
            best_match = chr(65 + i)  # 'A', 'B', 'C', or 'D'
            best_match_ratio = similarity

    # If we found a good match (you can adjust the threshold)
    if best_match_ratio > 0.7:
        return best_match

    # If all else fails, return a random choice
    return random.choice(['A', 'B', 'C', 'D'])


def cvqa_process_results(doc, results):
    """
    Process individual results and return accuracy metric.
    """
    target = cvqa_doc_to_target(doc)
    pred = parse_multi_choice_response(results[0], doc['Options'])

    # Convert to numerical for passthrough data
    pred_numerical = {'A':  0, 'B': 1, 'C': 2, 'D': 3}[pred]

    # Calculate if this prediction is correct
    is_correct = 1.0 if pred == target else 0.0

    return {
        "accuracy": is_correct,
        "cvqa_passthrough": {
            "id": doc["ID"],
            "pred": pred_numerical,
            "target":  target
        }
    }


def cvqa_test_aggregation_result(results, args):
    """
    Aggregate results and compute final accuracy.
    This function computes the mean accuracy across all samples.
    """
    total = len(results)
    correct = 0
    letter_to_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

    for result in results:
        target_idx = letter_to_idx[result['target']]
        if result['pred'] == target_idx:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0

    print(f"\n{'='*50}")
    print(f"CVQA Evaluation Results:")
    print(f"Total samples: {total}")
    print(f"Correct predictions:  {correct}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"{'='*50}\n")

    return accuracy