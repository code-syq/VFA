import os
import subprocess

# ==========================================
# 1. 基础路径配置 (只需要改这里)
# ==========================================

# 输入你的模型路径
VLM_MODEL_PATH = os.environ.get("VLM_MODEL_PATH", "/path/to/llama3-llava-next-8b-hf")
LLM_MODEL_PATH = os.environ.get("LLM_MODEL_PATH", "/path/to/finetuned-llama3-8b")

# 如果跑 Task Arithmetic/Ties 需要 Base
BASE_MODEL_PATH = os.environ.get("BASE_MODEL_PATH", "/path/to/Meta-Llama-3-8B-Instruct")

# 模型类型 (传给脚本的参数)
VLM_MODEL_TYPE = "llava-hf/llama3-llava-next-8b-hf"
LLM_MODEL_TYPE = "meta-llama/Meta-Llama-3-8B-Instruct"

# 结果保存的根目录
OUTPUT_ROOT = "outputs/merged_mllms"

# 执行脚本路径
SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vlm_model_merge.py")

# ==========================================
# 2. [自动提取] 从路径获取模型简称
# ==========================================
# 逻辑：去除末尾可能存在的斜杠，然后提取最后一个文件夹名
# 例如: /.../Idefics3-8B-Llama3/ -> Idefics3-8B-Llama3

VLM_NAME_FOR_FILE = os.path.basename(VLM_MODEL_PATH.rstrip('/'))
LLM_NAME_FOR_FILE = os.path.basename(LLM_MODEL_PATH.rstrip('/'))

print(f"[AUTO-DETECT] VLM Name: {VLM_NAME_FOR_FILE}")
print(f"[AUTO-DETECT] LLM Name: {LLM_NAME_FOR_FILE}")

# ==========================================
# 3. 命名映射规则
# ==========================================

# 规则1：根据 mode 决定存储的子文件夹 (debug_xxx)
MODE_DIR_MAP = {
    "weighted_average": "debug_weighted_average",
    "task_arithmetic":  "debug_task_arithmetic",
    # "ties":             "debug_ties"
}

# 规则2：根据 mode 决定文件名里的缩写 (WA, TA, TIES)
MODE_ABBR_MAP = {
    "weighted_average": "WA",
    "task_arithmetic":  "TA",
    # "ties":             "TIES"
}

# ==========================================
# 4. 配置列表生成
# ==========================================
MERGE_CONFIGS = []

# # (A) Weighted Average
MERGE_CONFIGS += [
    {"mode": "weighted_average", "alpha": alpha, "density": "null"}
    for alpha in [0.5, 0.7, 0.9]
]

# (B) Task Arithmetic
MERGE_CONFIGS += [
    {"mode": "task_arithmetic", "alpha": alpha, "density": "null"}
    for alpha in [0.5, 0.7, 0.9, 1.0]
]

# # (C) Ties (Density 根据需要修改，例如 0.2)
# MERGE_CONFIGS += [
#     {"mode": "ties", "alpha": alpha, "density": 0.2}
#     for alpha in [0.5, 0.7, 0.9, 1.0]
# ]

# ==========================================
# 5. 执行主循环
# ==========================================

for config in MERGE_CONFIGS:
    mode = config["mode"]
    alpha = config["alpha"]
    density = config["density"]

    # 1. Determine the output subdirectory
    base_output_dir = os.path.join(OUTPUT_ROOT, MODE_DIR_MAP[mode])

    # 2. 自动拼接文件夹名
    # 格式: [VLM名]-merged-with-[LLM名]-[缩写]-[Alpha]
    folder_name = f"{VLM_NAME_FOR_FILE}-merged-with-{LLM_NAME_FOR_FILE}-{MODE_ABBR_MAP[mode]}-{alpha}"

    # 如果 Ties 模式有 density，加在名字后面
    if mode == "ties" and density != "null":
        folder_name += f"-Density-{density}"

    # 拼接最终完整路径
    final_output_dir = os.path.join(base_output_dir, folder_name)

    print(f"\n[INFO] ------------------------------------------------")
    print(f"[INFO] Mode: {mode} | Alpha: {alpha}")
    print(f"[INFO] Saving to: {final_output_dir}")

    # 构建命令
    cmd = [
        "python", SCRIPT_PATH,
        "--vlm_model_type", VLM_MODEL_TYPE,
        "--llm_model_type", LLM_MODEL_TYPE,
        "--vlm_model_path", VLM_MODEL_PATH,
        "--llm_model_path", LLM_MODEL_PATH,
        "--output_dir", final_output_dir,
        "--mode", mode,
        "--alpha", str(alpha),
    ]

    # 特殊处理：Task Arithmetic 和 Ties 需要 Base Model
    if mode in ["task_arithmetic", "ties"]:
        if not BASE_MODEL_PATH:
            print(f"[ERROR] Skipping {mode}: BASE_MODEL_PATH is not set!")
            continue
        cmd.extend(["--base_model_path", BASE_MODEL_PATH])

    # 特殊处理：Density
    if density != "null":
        cmd.extend(["--density", str(density)])

    # 执行
    try:
        os.makedirs(final_output_dir, exist_ok=True)
        subprocess.run(cmd, check=True)
        print(f"[SUCCESS] Completed: {folder_name}")
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] Error in {folder_name}: {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")

print("\nAll tasks finished.")
