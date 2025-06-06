import json

# Create the data structure
data = {}

# Generate entries for fineweb_0 to fineweb_10
for i in range(11):
    key = f"fineweb_{i}"
    filename = f"fineweb_chunk_{i:03d}.jsonl"
    
    data[key] = {
        "file_name": filename,
        "columns": {
            "prompt": "text"
        }
    }

# Save to JSON file
with open("/home/aiops/zhangfz/LLaMA-Factory-Muon/dataset_self/fineweb/dataset_info.json", "w") as f:
    json.dump(data, f, indent=4)

print("JSON file created successfully!")