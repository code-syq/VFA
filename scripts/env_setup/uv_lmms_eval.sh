# !/bin/bash
set -ex

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv uv_lmms_eval --python 3.11
source uv_lmms_eval/bin/activate
uv pip install --upgrade pip
export UV_LINK_MODE=copy

cd src/lmms-eval
uv pip install -e ".[all]"

uv pip install vllm==0.13.0
uv pip install transformers==4.57.3
uv pip install langdetect
uv pip install openai azure.identity
uv pip install --upgrade --force-reinstall "protobuf==3.20.3"
uv pip install qwen-vl-utils
uv pip install ipdb
uv pip install jieba
uv pip install distance
uv pip install editdistance
uv pip install python-Levenshtein
uv pip install apted
uv pip install lxml
uv pip install spacy


uv pip list