#!/usr/bin/env python3
import argparse
import base64
import csv
import io
import json
import os
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset, DatasetDict, load_dataset
from openai import AzureOpenAI
from PIL import Image
from tqdm import tqdm

PROMPT = (
    "Describe this image in extreme detail. Include all numbers, geometric "
    "relationships, axis labels, and text visible in the image. Do not solve "
    "the problem. Only describe the visual information."
)

SPLIT_TO_DATASET_NAME = {
    "text_only": "testmini_text_only",
    "text_lite": "testmini_version_split",
    "text_dominant": "testmini_version_split",
    "vision_intensive": "testmini_version_split",
    "vision_dominant": "testmini_version_split",
    "vision_only": "testmini_version_split",
}


def _to_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


class AzureVisionCaptioner:
    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        endpoint: str = "",
        api_version: str = "",
        api_key: str = "",
        use_aad: bool = True,
        tenant_id: str = "",
    ):
        endpoint = (
            endpoint
            or os.getenv("AZURE_ENDPOINT")
            or os.getenv("AZURE_OPENAI_API_BASE")
            or os.getenv("AZURE_OPENAI_ENDPOINT")
            or "https://conversationhubeastus.openai.azure.com/"
        )

        api_version = api_version or os.getenv("AZURE_API_VERSION") or os.getenv("API_VERSION") or "2024-02-15-preview"
        api_key = api_key or os.getenv("AZURE_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY") or ""
        ad_token = os.getenv("AZURE_OPENAI_AD_TOKEN", "")
        use_aad = use_aad if use_aad is not None else _to_bool(os.getenv("AZURE_USE_AAD", "1"))
        if not tenant_id:
            tenant_id = os.getenv("AZURE_TENANT_ID", "")

        kwargs = {"azure_endpoint": endpoint.rstrip("/"), "api_version": api_version}
        if api_key:
            kwargs["api_key"] = api_key
        elif ad_token:
            kwargs["azure_ad_token"] = ad_token
        elif use_aad:
            try:
                from azure.identity import AzureCliCredential, get_bearer_token_provider

                cred = AzureCliCredential(tenant_id=tenant_id) if tenant_id else AzureCliCredential()
                kwargs["azure_ad_token_provider"] = get_bearer_token_provider(
                    cred, "https://cognitiveservices.azure.com/.default"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to init Azure AAD credential: {e}") from e
        else:
            raise ValueError(
                "Azure auth missing. Provide --api-key/AZURE_API_KEY, or set AZURE_OPENAI_AD_TOKEN, "
                "or keep --use-aad and run `az login`."
            )

        self.client = AzureOpenAI(**kwargs)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @staticmethod
    def _image_to_data_url(img: Image.Image) -> str:
        import io

        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    def caption(self, image: Image.Image, prompt: str) -> str:
        content = [
            {"type": "image_url", "image_url": {"url": self._image_to_data_url(image), "detail": "low"}},
            {"type": "text", "text": prompt},
        ]
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
        }
        if "gpt-5" in self.model:
            payload["max_completion_tokens"] = self.max_tokens
            if float(self.temperature) == 1.0:
                payload["temperature"] = self.temperature
        else:
            payload["max_tokens"] = self.max_tokens
            payload["temperature"] = self.temperature

        response = self.client.chat.completions.create(**payload)
        return (response.choices[0].message.content or "").strip()


def _sample_key(row: dict, fallback_idx: int) -> str:
    for key in ("sample_index", "problem_index", "index", "id"):
        if key in row:
            return str(row[key])
    return str(fallback_idx)


def _load_progress(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    data: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = str(obj["key"])
            caption = obj.get("caption", "")
            if isinstance(caption, str) and caption.strip():
                data[key] = caption
    return data


def _append_progress(path: Path, key: str, caption: str) -> None:
    record = {"key": key, "caption": caption}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _build_query(original_query_wo: str, caption: str, mode: str) -> str:
    if mode == "replace":
        return caption
    return (
        "Image description (generated from the original figure):\n"
        f"{caption}\n\n"
        "Question:\n"
        f"{original_query_wo}"
    )


def _decode_data_url(data_url: str) -> bytes:
    _, b64 = data_url.split(",", 1)
    return base64.b64decode(b64)


def _ensure_pil_image(image_obj: Any) -> Image.Image:
    if image_obj is None:
        raise ValueError("image is None")
    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")
    if isinstance(image_obj, list):
        if not image_obj:
            raise ValueError("image list is empty")
        return _ensure_pil_image(image_obj[0])
    if isinstance(image_obj, dict):
        if image_obj.get("bytes"):
            return Image.open(io.BytesIO(image_obj["bytes"])).convert("RGB")
        if image_obj.get("path"):
            return _ensure_pil_image(image_obj["path"])
        raise ValueError(f"unsupported image dict keys: {list(image_obj.keys())}")
    if isinstance(image_obj, bytes):
        return Image.open(io.BytesIO(image_obj)).convert("RGB")
    if isinstance(image_obj, str):
        s = image_obj.strip()
        if not s:
            raise ValueError("image string is empty")
        if s.startswith("data:image/"):
            return Image.open(io.BytesIO(_decode_data_url(s))).convert("RGB")
        if s.startswith(("http://", "https://")):
            with urllib.request.urlopen(s, timeout=30) as r:
                return Image.open(io.BytesIO(r.read())).convert("RGB")
        return Image.open(s).convert("RGB")
    raise TypeError(f"unsupported image type: {type(image_obj)}")


def _serialize_cell(v: Any) -> str:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return "" if v is None else str(v)
    if isinstance(v, bytes):
        return f"<bytes:{len(v)}>"
    if isinstance(v, Image.Image):
        return f"<PIL.Image mode={v.mode} size={v.size}>"
    try:
        return json.dumps(v, ensure_ascii=False)
    except TypeError:
        return str(v)


def _write_split_exports(split_dir: Path, rows: List[dict]) -> None:
    jsonl_path = split_dir / "samples.jsonl"
    csv_path = split_dir / "samples.csv"

    normalized_rows: List[Dict[str, str]] = []
    columns: List[str] = []
    seen = set()
    for row in rows:
        nr = {k: _serialize_cell(v) for k, v in row.items()}
        normalized_rows.append(nr)
        for k in nr.keys():
            if k not in seen:
                seen.add(k)
                columns.append(k)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for nr in normalized_rows:
            f.write(json.dumps(nr, ensure_ascii=False) + "\n")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(normalized_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MathVerse blind dataset for lmms-eval.")
    parser.add_argument("--dataset-path", default="CaraJ/MathVerse-lmmseval")
    parser.add_argument(
        "--splits",
        default="text_only,text_lite,text_dominant,vision_intensive,vision_dominant,vision_only",
    )
    parser.add_argument("--out-dir", required=True, help="Output dir for datasets.save_to_disk.")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--query-mode", choices=["append", "replace"], default="append")
    parser.add_argument("--caption-field", default="blind_caption")
    parser.add_argument(
        "--blind-query-field",
        default="blind_query_wo",
        help="Field name for caption-augmented query (original query_wo is kept unchanged).",
    )
    parser.add_argument(
        "--blind-query-cot-field",
        default="blind_query_cot",
        help="Field name for caption-augmented query_cot.",
    )
    parser.add_argument(
        "--figure-note-field",
        default="blind_figure_notice",
        help="Field name that stores note about figure being unavailable.",
    )
    parser.add_argument(
        "--figure-note",
        default=(
            "Note: The original figure is unavailable in this setting. "
            "Do not rely on visual access to the figure. Use the provided textual image description instead."
        ),
        help="Instruction note inserted as an extra dataset field.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--endpoint", default="", help="Azure endpoint. Defaults to env vars, then eastus public endpoint.")
    parser.add_argument("--api-version", default="", help="Azure API version. Defaults to AZURE_API_VERSION/API_VERSION/2024-02-15-preview.")
    parser.add_argument("--api-key", default="", help="Azure API key. Optional if using AAD.")
    parser.add_argument("--use-aad", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--tenant-id", default="", help="Optional AAD tenant id when using Azure CLI credential.")
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit per split.")
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--drop-image", action="store_true")
    parser.add_argument(
        "--blind-image",
        default="",
        help="Optional local image path. If set, replace each sample image with this blank image (for blind vision eval).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    unknown = [s for s in splits if s not in SPLIT_TO_DATASET_NAME]
    if unknown:
        raise ValueError(f"Unsupported splits: {unknown}. Supported: {list(SPLIT_TO_DATASET_NAME.keys())}")
    if args.blind_image and not Path(args.blind_image).exists():
        raise FileNotFoundError(f"--blind-image not found: {args.blind_image}")
    blind_image = Image.open(args.blind_image).convert("RGB") if args.blind_image else None

    out_dir = Path(args.out_dir).resolve()
    progress_dir = out_dir / "_progress"
    progress_dir.mkdir(parents=True, exist_ok=True)

    if args.use_aad == "auto":
        use_aad = _to_bool(os.getenv("AZURE_USE_AAD", "1"))
    else:
        use_aad = args.use_aad == "true"

    api = AzureVisionCaptioner(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        endpoint=args.endpoint,
        api_version=args.api_version,
        api_key=args.api_key,
        use_aad=use_aad,
        tenant_id=args.tenant_id,
    )

    ds_out = DatasetDict()
    with tempfile.TemporaryDirectory(prefix="mathverse_blind_") as tmpdir:
        tmp_path = Path(tmpdir)
        for split in splits:
            dataset_name = SPLIT_TO_DATASET_NAME[split]
            print(f"[{split}] loading: path={args.dataset_path}, name={dataset_name}, split={split}")
            ds = load_dataset(args.dataset_path, name=dataset_name, split=split)
            if args.limit and args.limit > 0:
                ds = ds.select(range(min(args.limit, len(ds))))

            rows: List[dict] = [dict(row) for row in ds]
            progress_file = progress_dir / f"{split}.jsonl"
            cache = _load_progress(progress_file)
            print(f"[{split}] loaded {len(rows)} rows, cached captions: {len(cache)}")

            done = 0
            pbar = tqdm(total=len(rows), desc=f"{split} captioning", dynamic_ncols=True, mininterval=1.0)
            for i, row in enumerate(rows):
                key = _sample_key(row, i)
                prev = cache.get(key, "")
                if prev and (not args.retry_failed or not prev.startswith("Failed to obtain answer")):
                    caption = prev
                else:
                    image_obj = row.get("image", None)
                    if image_obj is None:
                        caption = "Failed to obtain answer via API. Missing image field."
                    else:
                        try:
                            image_file = tmp_path / f"{split}_{key}.jpg"
                            rgb_image = _ensure_pil_image(image_obj)
                            rgb_image.save(image_file)
                            caption = api.caption(rgb_image, args.prompt)
                        except Exception as e:
                            caption = f"Failed to obtain answer via API. Image decode error: {e}"
                    _append_progress(progress_file, key, caption)

                original_query_wo = str(row.get("query_wo", row.get("question", "")))
                original_query_cot = str(row.get("query_cot", original_query_wo))
                row[args.caption_field] = caption
                row[args.figure_note_field] = args.figure_note
                row[args.blind_query_field] = _build_query(original_query_wo, caption, args.query_mode)
                row[args.blind_query_cot_field] = _build_query(original_query_cot, caption, args.query_mode)
                row["blind_mode"] = args.query_mode

                if blind_image is not None and "image" in row and row["image"] is not None:
                    row["image"] = blind_image.copy()

                if args.drop_image and "image" in row:
                    row.pop("image", None)

                rows[i] = row
                done += 1
                pbar.update(1)

                if args.save_every > 0 and done % args.save_every == 0:
                    time.sleep(0.1)

            pbar.close()
            ds_out[split] = Dataset.from_list(rows)
            print(f"[{split}] finished: {len(rows)} rows")

    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output dir already exists: {out_dir}. Use --overwrite to replace it.")
        # save_to_disk refuses to overwrite existing directory.
        import shutil

        shutil.rmtree(out_dir)
    ds_out.save_to_disk(str(out_dir))
    for split in ds_out.keys():
        split_dir = out_dir / split
        split_rows = [dict(x) for x in ds_out[split]]
        _write_split_exports(split_dir, split_rows)

    print("\nSaved blind dataset to:")
    print(out_dir)
    print("Also exported human-readable files in each split dir: samples.csv, samples.jsonl")
    print("\nUse with lmms-eval task YAML:")
    print(f"dataset_path: {out_dir}")
    print("dataset_kwargs:")
    print("  load_from_disk: true")


if __name__ == "__main__":
    main()
