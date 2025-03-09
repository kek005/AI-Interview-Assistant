import os

# Model and tokenizer paths
MODEL_NAME = "microsoft/phi-2"
OUTPUT_DIR = "./models/phi2_finetuned_azure"

# Data files
DATA_PATH = "./data/fine_tune_data.jsonl"

# Training parameters
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-4
EPOCHS = 3