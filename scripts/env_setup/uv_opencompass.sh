#!/bin/bash
set -ex

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv uv_opencompass --python 3.11
source uv_opencompass/bin/activate
uv pip install --upgrade pip
export UV_LINK_MODE=copy

uv pip install opencompass
uv pip install vllm
uv pip install flash-attn --no-build-isolation
uv pip list
