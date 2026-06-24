# !/bin/bash
set -ex

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv uv_pangea --python 3.11
source uv_pangea/bin/activate
uv pip install --upgrade pip
export UV_LINK_MODE=copy

cd src/Pangea/train/LLaVA-NeXT
uv pip install -e .

cd ../../evaluation/lmms-eval
uv pip install -e .

uv pip install vllm==0.13.0
uv pip install transformers==4.57.3
uv pip install langdetect
uv pip install openai azure.identity

# uv pip install sglang
uv pip install --upgrade --force-reinstall "protobuf==3.20.3"
uv pip install qwen-vl-utils

sudo apt-get update
sudo apt-get install -y openjdk-17-jre-headless

uv pip list