import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset

# Load the Phi model and tokenizer
MODEL_NAME = "microsoft/phi-2"  # Change if using a different Phi model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Load dataset (example: Hugging Face's "wikitext" dataset)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_data = dataset["train"].map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length"), batched=True)

# Data collator to pad inputs for training
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Set to True if using masked language modeling
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./phi_finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=100,
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=True if torch.cuda.is_available() else False,  # Enable mixed precision training if GPU available
    push_to_hub=False
)

# Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    data_collator=data_collator
)

# Start training
trainer.train()
