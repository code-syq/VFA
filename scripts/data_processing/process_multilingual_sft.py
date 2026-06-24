"""
Process the multilingual-sft dataset from HuggingFace.

Downloads agentlans/multilingual-sft (clustered_k100000 subset),
filters out samples containing multimodal tokens (<image>, <audio>, <video>),
and saves the cleaned data as JSON for LLaMA-Factory training.

Usage:
    python scripts/data_processing/process_multilingual_sft.py [--output_dir data]
"""
from datasets import load_dataset
import os
import json
import argparse


def write_json(dict_objs, file_name):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w+", encoding="utf-8") as f:
        json.dump(dict_objs, f, indent=4, ensure_ascii=False)


def contains_tokens(x):
    if not isinstance(x, str):
        return True
    if "<audio>" in x:
        return True
    if "<image>" in x:
        return True
    if "<video>" in x:
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Process multilingual-sft dataset")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Output directory (default: data)")
    args = parser.parse_args()

    # 1. Load multilingual-sft
    dataset = load_dataset(
        "agentlans/multilingual-sft",
        "clustered_k100000",
        split="train"
    )

    output_data = []

    # 2. Filter out multimodal samples
    for i in range(len(dataset)):
        item = dataset[i]

        input_text = item.get("input", "")
        source_text = item.get("source", "")

        if contains_tokens(input_text):
            continue
        if contains_tokens(source_text):
            continue

        output_data.append(item)

    # 3. Save
    num_samples = len(output_data)
    output_file = os.path.join(args.output_dir, f"multilingual_sft_{num_samples}.json")

    write_json(output_data, output_file)
    print(f"Saved {num_samples} samples to {output_file}")


if __name__ == "__main__":
    main()
