#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional


try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("Missing dependency: pyyaml. Please install it into uv_llamafactory.") from e


PYTHON_BIN = sys.executable
_PLACEHOLDER_RE = re.compile(r"\{([^{}]+)\}")


def _get_by_path(data: Mapping[str, Any], dotted_path: str) -> Any:
    cur: Any = data
    for part in dotted_path.split("."):
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        else:
            raise KeyError(f"Placeholder {{{dotted_path}}} not found (stuck at '{part}').")
    return cur


def _format_string(s: str, ctx: Mapping[str, Any]) -> str:
    def repl(m: re.Match[str]) -> str:
        key = m.group(1).strip()
        val = _get_by_path(ctx, key)
        # YAML `null` loads as Python None; for config/args we want the literal "null".
        if val is None:
            return "null"
        return str(val)

    return _PLACEHOLDER_RE.sub(repl, s)


def _deep_format(obj: Any, ctx: Mapping[str, Any]) -> Any:
    if isinstance(obj, str):
        return _format_string(obj, ctx)
    if isinstance(obj, list):
        return [_deep_format(x, ctx) for x in obj]
    if isinstance(obj, dict):
        return {k: _deep_format(v, ctx) for k, v in obj.items()}
    return obj


def load_and_expand_yaml(path: str, max_passes: int = 10) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping/dict, got: {type(data).__name__}")

    cur: Dict[str, Any] = data
    for _ in range(max_passes):
        nxt = _deep_format(cur, cur)
        if nxt == cur:
            return cur
        cur = nxt
    raise RuntimeError(f"Placeholder expansion did not converge after {max_passes} passes: {path}")


def _run(cmd: List[str], dry_run: bool) -> None:
    printable = " ".join(shlex.quote(x) for x in cmd)
    print(f"[CMD] {printable}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def run_merge_lora(lora_merge_config: Mapping[str, Any], dry_run: bool) -> None:
    if not lora_merge_config.get("need_lora_merge", False):
        print("[SKIP] lora_merge_config.need_lora_merge=false")
        return

    base_ckpt = str(lora_merge_config["base_ckpt"])
    lora_ckpt = str(lora_merge_config["lora_ckpt"])
    merged_ckpt = str(lora_merge_config["merged_ckpt"])
    _run(
        ["bash", "scripts/utils/merge_lora.sh", base_ckpt, lora_ckpt, merged_ckpt],
        dry_run=dry_run,
    )


def run_model_merge(model_merge_config: Mapping[str, Any], dry_run: bool) -> None:
    if not model_merge_config.get("need_model_merge", False):
        print("[SKIP] model_merge_config.need_model_merge=false")
        return

    def add_arg(argv: List[str], key: str, val: Any) -> None:
        # Keep real config values as-is: if it's None, omit the CLI arg.
        # Only placeholder substitution in `{...}` maps None -> "null".
        if val is None:
            return
        argv.extend([f"--{key}", str(val)])

    argv: List[str] = [PYTHON_BIN, "scripts/merge/vlm_model_merge.py"]
    add_arg(argv, "vlm_model_type", model_merge_config["vlm_model_type"])
    add_arg(argv, "llm_model_type", model_merge_config["llm_model_type"])
    add_arg(argv, "vlm_model_path", model_merge_config["vlm_model_path"])
    add_arg(argv, "llm_model_path", model_merge_config["llm_model_path"])
    add_arg(argv, "base_model_path", model_merge_config.get("base_model_path"))
    add_arg(argv, "output_dir", model_merge_config["output_dir"])
    add_arg(argv, "mode", model_merge_config.get("mode", "base"))
    add_arg(argv, "alpha", model_merge_config["alpha"])
    add_arg(argv, "density", model_merge_config.get("density"))
    add_arg(argv, "alpha2", model_merge_config.get("alpha2"))
    if model_merge_config.get("no_progress", False):
        argv.append("--no_progress")

    _run(argv, dry_run=dry_run)


def run_eval(eval_config: Mapping[str, Any], dry_run: bool) -> None:
    eval_ckpt = str(eval_config["eval_ckpt"])
    eval_output_root = str(eval_config.get("eval_output_root", "outputs/eval_results"))
    _run(
        ["bash", "scripts/eval/eval_lmms.sh", eval_ckpt, eval_output_root],
        dry_run=dry_run,
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run debug_config.yaml pipeline (merge lora -> merge model -> eval).")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to pipeline_config.yaml",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing them.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = load_and_expand_yaml(args.config)
    run_merge_lora(cfg.get("lora_merge_config", {}) or {}, dry_run=args.dry_run)
    run_model_merge(cfg.get("model_merge_config", {}) or {}, dry_run=args.dry_run)
    run_eval(cfg.get("eval_config", {}) or {}, dry_run=args.dry_run)

if __name__ == "__main__":
    main()