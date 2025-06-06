from datasets import load_dataset
import json
import os

# Configuration
dataset_name = "HuggingFaceFW/fineweb"
config_name = "sample-10BT"
split = "train"
output_dir = "/home/aiops/zhangfz/LLaMA-Factory-Muon/dataset_self/fineweb"
chunk_size = 10000
max_chunks = 10  # Set to None to stream everything

os.makedirs(output_dir, exist_ok=True)

# Load streamed dataset
dataset = load_dataset(dataset_name, name=config_name, split=split, streaming=True)

file_index = 0
line_count = 0
output_file = None

for i, example in enumerate(dataset):
    if line_count % chunk_size == 0:
        if output_file:
            output_file.close()
        filename = os.path.join(output_dir, f"fineweb_chunk_{file_index:03d}.jsonl")
        output_file = open(filename, "w", encoding="utf-8")
        print(f"Writing to {filename}")
        file_index += 1
        if max_chunks is not None and file_index > max_chunks:
            break

    json_line = json.dumps({"text": example["text"]}, ensure_ascii=False)
    output_file.write(json_line + "\n")
    line_count += 1

if output_file:
    output_file.close()
