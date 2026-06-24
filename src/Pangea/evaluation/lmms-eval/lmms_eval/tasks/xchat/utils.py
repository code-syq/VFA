import json
import os
import sys
import importlib.util
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import requests
import yaml
from PIL import Image
from datasets import Dataset
from langdetect import detect
from loguru import logger as eval_logger

NUM_SECONDS_TO_SLEEP = 5
BASE_DIR = Path(__file__).resolve().parent
PROMPT_DIR = BASE_DIR / "prompts"

XCHAT_CATEGORY_TO_METRIC = {
    "science_figure_explanation": "gpt_eval_xchat_science_figure_explanation",
    "ocr": "gpt_eval_xchat_ocr",
    "defeasible_reasoning": "gpt_eval_xchat_defeasible_reasoning",
    "iq_test": "gpt_eval_xchat_iq_test",
    "art_explanation": "gpt_eval_xchat_art_explanation",
    "image_humor_understanding": "gpt_eval_xchat_image_humor_understanding",
    "unusual_images": "gpt_eval_xchat_unusual_images",
    "graph_interpretation": "gpt_eval_xchat_graph_interpretation",
    "cultural_understanding": "gpt_eval_xchat_cultural_understanding",
}

XCHAT_METRICS = list(XCHAT_CATEGORY_TO_METRIC.values())

with open(BASE_DIR / "_default_template.yaml", "r", encoding="utf-8") as f:
    raw_data = f.readlines()
    safe_data = [line for line in raw_data if "!function" not in line]
    CONFIG = yaml.safe_load("".join(safe_data))

GPT_EVAL_MODEL_NAME = CONFIG["metadata"]["gpt_eval_model_name"]

API_TYPE = os.getenv("API_TYPE", "openai")
API_URL = None
API_KEY = None
headers: Dict[str, str] = {}

if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
elif API_TYPE == "azure":
    API_URL = os.getenv("AZURE_ENDPOINT", "https://api.cognitive.microsoft.com/sts/v1.0/issueToken")
    API_KEY = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")
    headers = {
        "api-key": API_KEY,
        "Content-Type": "application/json",
    }
elif API_TYPE == "azure_msra":
    # Use Azure MSRA endpoints defined in `azure_msra.py`.
    # The actual client will be created lazily in `get_eval`, so no headers/API_URL needed here.
    pass
else:
    raise ValueError(f"Unsupported API_TYPE: {API_TYPE}")

_LANG_DIR_CACHE: Dict[str, str] = {}
_METADATA_CACHE: Dict[Tuple[str, str], Dict[int, Dict[str, Any]]] = {}


def _discover_language_directory(lang_key: str) -> str:
    lang_key = lang_key.lower()
    if lang_key in _LANG_DIR_CACHE:
        return _LANG_DIR_CACHE[lang_key]

    for child in BASE_DIR.iterdir():
        if child.is_dir() and child.name not in {"prompts", "responses", "__pycache__"}:
            if child.name.lower() == lang_key:
                _LANG_DIR_CACHE[lang_key] = child.name
                return child.name

    raise FileNotFoundError(f"Cannot locate directory for language '{lang_key}' under {BASE_DIR}")


def _resolve_image_components(image_path: str) -> Tuple[str, str, int, Path]:
    path = Path(image_path)
    language_part = None
    category = None

    if "xchat" in path.parts:
        idx = path.parts.index("xchat")
        if idx + 2 < len(path.parts):
            language_part = path.parts[idx + 1]
            category = path.parts[idx + 2]
    if language_part is None or category is None:
        try:
            language_part = path.parent.parent.name
            category = path.parent.name
        except Exception:  # noqa: BLE001
            raise ValueError(f"Unable to parse language/category from image path: {image_path}")

    filename = path.name
    language_dir = _discover_language_directory(language_part)
    resolved_path = BASE_DIR / language_dir / category / filename
    if not resolved_path.exists():
        raise FileNotFoundError(f"Resolved image path does not exist: {resolved_path}")

    try:
        instance_idx = int(Path(filename).stem)
    except ValueError as exc:
        raise ValueError(f"Expected numeric filename but received: {filename}") from exc

    return language_dir, category, instance_idx, resolved_path


def _load_category_metadata(language_dir: str, category: str) -> Dict[int, Dict[str, Any]]:
    cache_key = (language_dir, category)
    if cache_key in _METADATA_CACHE:
        return _METADATA_CACHE[cache_key]

    data_path = BASE_DIR / language_dir / category / "data.json"
    if not data_path.exists():
        _METADATA_CACHE[cache_key] = {}
        return _METADATA_CACHE[cache_key]

    with data_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    category_metadata = {record.get("instance_idx", idx): record for idx, record in enumerate(records)}
    _METADATA_CACHE[cache_key] = category_metadata
    return category_metadata


def _gather_metadata(image_path: str) -> Dict[str, Any]:
    language_dir, category, instance_idx, resolved_path = _resolve_image_components(image_path)
    metadata_entries = _load_category_metadata(language_dir, category)
    metadata = metadata_entries.get(instance_idx, {}).copy()
    metadata.update(
        {
            "language": language_dir,
            "category": category,
            "instance_idx": instance_idx,
            "resolved_image_path": resolved_path.as_posix(),
        }
    )
    return metadata


def xchat_process_docs(dataset: Dataset) -> Dataset:
    def _augment(example: Dict[str, Any]) -> Dict[str, Any]:
        metadata = _gather_metadata(example.get("image", ""))
        system_prompt = metadata.get("system_prompt", "")
        user_input = metadata.get("input") or example.get("text", "")
        example.update(
            {
                "language": metadata.get("language", ""),
                "task": metadata.get("task", metadata.get("category", "")),
                "system_prompt": system_prompt,
                "input": user_input,
                "reference_answer": metadata.get("reference_answer", ""),
                "score_rubric": metadata.get("score_rubric", {}),
                "atomic_checklist": metadata.get("atomic_checklist", []),
                "background_knowledge": metadata.get("background_knowledge", []),
                "resolved_image_path": metadata.get("resolved_image_path", example.get("image", "")),
                "caption": metadata.get("caption", ""),
                "capability": metadata.get("capability", "vision"),
                "instance_idx": metadata.get("instance_idx"),
            }
        )
        # Normalize image field so downstream consumers rely on the resolved path
        example["image"] = example.get("resolved_image_path", example.get("image", ""))
        return example

    return dataset.map(_augment)


def _ensure_augmented_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    enriched = dict(doc)
    needs_metadata = not enriched.get("resolved_image_path") or not enriched.get("language")
    image_hint = enriched.get("resolved_image_path") or enriched.get("image")

    if needs_metadata and image_hint:
        try:
            metadata = _gather_metadata(image_hint)
        except Exception as exc:  # noqa: BLE001
            eval_logger.debug(f"Falling back to raw image path for doc augmentation: {exc}")
            metadata = {}
        for key, value in metadata.items():
            enriched.setdefault(key, value)

    if not enriched.get("resolved_image_path") and image_hint:
        enriched["resolved_image_path"] = image_hint

    if not enriched.get("input"):
        enriched["input"] = enriched.get("text", "")

    return enriched


def xchat_doc_to_visual(doc):
    augmented = _ensure_augmented_doc(doc)
    image_path_str = augmented.get("resolved_image_path")
    if not image_path_str:
        raise ValueError("Image path missing for xchat instance")
    image_path = Path(image_path_str)
    if not image_path.is_absolute():
        image_path = (BASE_DIR / image_path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Resolved image path does not exist: {image_path}")
    return [Image.open(image_path).convert("RGB")]


def xchat_doc_to_text(doc, model_specific_prompt_kwargs=None):
    augmented = _ensure_augmented_doc(doc)
    if model_specific_prompt_kwargs is None:
        model_specific_prompt_kwargs = {}
    pre_prompt = model_specific_prompt_kwargs.get("pre_prompt", "")
    post_prompt = model_specific_prompt_kwargs.get("post_prompt", "")

    parts = []
    system_prompt = (augmented.get("system_prompt") or "").strip()
    user_input = (augmented.get("input") or augmented.get("text") or "").strip()

    if system_prompt:
        parts.append(system_prompt)
    if user_input:
        parts.append(user_input)

    body = "\n\n".join(parts)
    return f"{pre_prompt}{body}{post_prompt}"


def xchat_doc_to_target(doc):
    augmented = _ensure_augmented_doc(doc)
    return augmented.get("reference_answer", "")


def _format_rubric(score_rubric: Dict[str, Any]) -> str:
    if not score_rubric:
        return "No explicit rubric provided."
    lines = [f"- Criteria: {score_rubric.get('criteria', '').strip()}"]
    for score in range(1, 6):
        key = f"score{score}_description"
        if key in score_rubric:
            lines.append(f"- Score {score}: {score_rubric[key]}")
    return "\n".join(lines)


def _format_checklist(checklist) -> str:
    if not checklist:
        return "None"
    return "\n".join(f"- {item}" for item in checklist)


def _format_background(background):
    if not background:
        return "None"
    lines = []
    for entry in background:
        info = (entry.get("information") or "").strip()
        url = (entry.get("reference_url") or "").strip()
        if info and url:
            lines.append(f"- {info} (Source: {url})")
        elif url:
            lines.append(f"- Source: {url}")
        elif info:
            lines.append(f"- {info}")
    return "\n".join(lines) if lines else "None"


def _build_eval_prompt(doc, candidate_answer: str) -> str:
    augmented = _ensure_augmented_doc(doc)
    system_prompt = augmented.get("system_prompt", "")
    user_input = augmented.get("input", "")
    reference_answer = augmented.get("reference_answer", "")
    caption = augmented.get("caption", "")
    capability = augmented.get("capability", "")
    rubric = _format_rubric(augmented.get("score_rubric", {}))
    checklist = _format_checklist(augmented.get("atomic_checklist", []))
    background = _format_background(augmented.get("background_knowledge", []))

    sections = [
        "You are an impartial grader for multilingual, multimodal tasks.",
        "Use the rubric and checklist to score both answers from 1 to 5.",
        "Return the first line as two numbers separated by a space: '<reference_score> <candidate_score>'.",
        "Follow the scores with a concise justification referencing the rubric.",
        "If the candidate answer is in a different language from both the question and the reference answer, reflect this penalty in the score.",
        "\n[Capability]\n" + capability,
        "[System Prompt]\n" + system_prompt,
        "[User Instruction]\n" + user_input,
        "[Optional Caption]\n" + caption,
        "[Background Knowledge]\n" + background,
        "[Rubric]\n" + rubric,
        "[Checklist]\n" + checklist,
        "[Reference Answer]\n" + reference_answer,
        "[Candidate Answer]\n" + candidate_answer,
    ]
    return "\n\n".join(section for section in sections if section.strip())


def get_eval(content: str, max_tokens: int, retries: int = 5):
    global headers

    messages = [
        {
            "role": "system",
            "content": "You are a helpful and precise assistant for checking the quality of the answer.",
        },
        {"role": "user", "content": content},
    ]

    payload = {
        "model": GPT_EVAL_MODEL_NAME,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }

    # Special handling for API_TYPE == "azure_msra", which uses the AzureOpenAI client
    if API_TYPE == "azure_msra":
        try:
            module_name = "azure_msra"
            module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "azure_msra.py")
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            azure_msra = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = azure_msra
            spec.loader.exec_module(azure_msra)
        except ImportError as e:
            eval_logger.error(f"Failed to import azure_msra when API_TYPE is 'azure_msra': {e}")
            return "", ""

        for attempt in range(retries):
            try:
                client, resolved_model = azure_msra.get_client(model_name=GPT_EVAL_MODEL_NAME)
                response = client.chat.completions.create(
                    model=resolved_model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content.strip()
                if content != "":
                    return content, resolved_model
                break
            except Exception as e:
                eval_logger.info(f"Attempt {attempt + 1} (azure_msra) failed with error: {e}")
                if attempt < retries:
                    time.sleep(NUM_SECONDS_TO_SLEEP)
                else:
                    eval_logger.error(f"All {retries} attempts failed for azure_msra. Last error message: {e}")
                    return "", ""
        return "", ""

    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            response_data = response.json()
            content = response_data["choices"][0]["message"]["content"].strip()
            if content:
                return content, response_data.get("model", GPT_EVAL_MODEL_NAME)
            break
        except Exception as exc:  # noqa: BLE001
            eval_logger.info(f"Attempt {attempt + 1} failed with error: {exc}")
            if attempt < retries - 1:
                time.sleep(NUM_SECONDS_TO_SLEEP)
            else:
                eval_logger.error(f"All {retries} attempts failed. Last error message: {exc}")
                return "", ""
    return "", ""


def parse_score(review: str):
    try:
        score_pair = review.split("\n")[0].replace(",", " ").split()
        if len(score_pair) != 2:
            raise ValueError(f"Unexpected score format: {review}")
        return [float(score_pair[0]), float(score_pair[1])]
    except Exception as exc:  # noqa: BLE001
        eval_logger.debug(f"Failed to parse score from review '{review}': {exc}")
        return [-1.0, -1.0]


def xchat_process_results(doc, result):
    try:
        augmented = _ensure_augmented_doc(doc)
        candidate_answer = result[0] if result else ""
        review_prompt = _build_eval_prompt(augmented, candidate_answer)
        review, model_name = get_eval(review_prompt, 1024)
        scores = parse_score(review)

        reference_answer = augmented.get("reference_answer", "")
        candidate_language = detect(candidate_answer) if candidate_answer.strip() else ""
        reference_language = detect(reference_answer) if reference_answer.strip() else ""
        question_language = detect(augmented.get("input", "")) if augmented.get("input") else ""

        if candidate_language and candidate_language not in {reference_language, question_language}:
            scores[1] = min(0.0, scores[1])

    except Exception as exc:  # noqa: BLE001
        eval_logger.error(f"Error grading instance {doc.get('instance_idx')}: {exc}")
        review = "Failed to get a proper review."
        model_name = "Failed Request"
        scores = [-1.0, -1.0]

    category = augmented.get("task", augmented.get("category", "all"))
    metric_name = XCHAT_CATEGORY_TO_METRIC.get(category, "gpt_eval_xchat_all")

    review_payload = {
        "system_prompt": augmented.get("system_prompt", ""),
        "instruction": augmented.get("input", augmented.get("text", "")),
        "reference_answer": augmented.get("reference_answer", ""),
        "candidate_answer": result[0] if result else "",
        "review": review,
        "scores": scores,
        "eval_model": model_name,
        "category": category,
        "language": augmented.get("language", ""),
        "rubric": augmented.get("score_rubric", {}),
        "checklist": augmented.get("atomic_checklist", []),
        "background": augmented.get("background_knowledge", []),
    }

    placeholder_payload = deepcopy(review_payload)
    placeholder_payload["scores"] = [-999.0, -999.0]

    data_dict = {}
    for metric in XCHAT_METRICS:
        data_dict[metric] = review_payload if metric == metric_name else placeholder_payload
    data_dict["gpt_eval_xchat_all"] = review_payload
    return data_dict


def _compute_percentage(scores_list):
    try:
        filtered = [scores for scores in scores_list if -999.0 not in scores["scores"]]
        if not filtered:
            return None
        scores_array = np.asarray([entry["scores"] for entry in filtered])
        ref_avg = float(scores_array[:, 0].mean())
        model_avg = float(scores_array[:, 1].mean())
        if ref_avg <= 0:
            return None
        return round(model_avg / ref_avg * 100, 1)
    except Exception as exc:  # noqa: BLE001
        eval_logger.info(f"Error computing aggregation: {exc}")
        return None


def xchat_aggregation(results, category):
    return _compute_percentage(results)


def xchat_all_aggregation(results):
    return xchat_aggregation(results, "all")


def xchat_science_figure_explanation_aggregation(results):
    return xchat_aggregation(results, "science_figure_explanation")


def xchat_ocr_aggregation(results):
    return xchat_aggregation(results, "ocr")


def xchat_defeasible_reasoning_aggregation(results):
    return xchat_aggregation(results, "defeasible_reasoning")


def xchat_iq_test_aggregation(results):
    return xchat_aggregation(results, "iq_test")


def xchat_art_explanation_aggregation(results):
    return xchat_aggregation(results, "art_explanation")


def xchat_image_humor_understanding_aggregation(results):
    return xchat_aggregation(results, "image_humor_understanding")


def xchat_unusual_images_aggregation(results):
    return xchat_aggregation(results, "unusual_images")


def xchat_graph_interpretation_aggregation(results):
    return xchat_aggregation(results, "graph_interpretation")


def xchat_cultural_understanding_aggregation(results):
    return xchat_aggregation(results, "cultural_understanding")


if __name__ == "__main__":
    # Quick sanity check:
    #   export API_TYPE="azure_msra"
    #   export API_TYPE="openai"
    test_content = "Reference answer score 5, candidate answer score 4."
    print(f"API_TYPE: {API_TYPE}")
    print(f"GPT_EVAL_MODEL_NAME: {GPT_EVAL_MODEL_NAME}")
    review, model_name = get_eval(test_content, max_tokens=64)
    print("Review:", review)
    print("Eval model name:", model_name)