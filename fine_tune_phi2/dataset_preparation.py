import csv
import json
from datasets import load_dataset

def csv_to_jsonl(csv_file, jsonl_file):
    """Converts CSV to JSONL for fine-tuning."""
    data = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry = {
                "prompt": row["Question"].strip(),
                "completion": row["Answer"].strip()
            }
            data.append(entry)
    
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
    
    print(f"[INFO] Successfully converted {csv_file} to {jsonl_file}")

def load_data(jsonl_file):
    """Loads dataset for fine-tuning and ensures it is formatted correctly."""
    dataset = load_dataset("json", data_files=jsonl_file, split="train")
    
    # Check if dataset is correctly loaded
    print(f"[DEBUG] Sample Data: {dataset[0]}")  # ✅ Print the first row for debugging

    return dataset