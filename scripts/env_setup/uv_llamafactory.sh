#!/bin/bash
set -ex

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv uv_llamafactory --python 3.11
source uv_llamafactory/bin/activate
uv pip install --upgrade pip
export UV_LINK_MODE=copy

cd src/LLaMA-Factory
uv pip install setuptools
uv pip install -e ".[torch,metrics]" --no-build-isolation
uv pip install "transformers==4.53.0" "tokenizers>=0.21.0,<0.22"
uv pip install deepspeed==0.16.9
uv pip install wandb
uv pip install hf_transfer
uv pip list
