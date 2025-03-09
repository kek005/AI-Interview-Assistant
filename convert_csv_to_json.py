import csv
import json

def csv_to_jsonl(csv_file, jsonl_file):
    """Converts a CSV file with Question-Answer pairs into JSONL format for fine-tuning."""
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

# Example usage
csv_to_jsonl("data/fine_tune_qa.csv", "data/fine_tune_data.jsonl")