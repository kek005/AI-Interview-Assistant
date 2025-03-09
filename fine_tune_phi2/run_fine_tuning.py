from dataset_preparation import csv_to_jsonl
from training import train
from evaluation import evaluate

# Convert CSV to JSONL
csv_to_jsonl("./data/fine_tune_qa.csv", "./data/fine_tune_data.jsonl")

# Fine-tune the model
train()

# Evaluate the model
#evaluate()