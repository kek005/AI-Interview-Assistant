from transformers import TrainingArguments, Trainer
from dataset_preparation import load_data
from model_loading import load_model
from config import OUTPUT_DIR, DATA_PATH, BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS, LEARNING_RATE, EPOCHS

def train():
    """Fine-tunes the Phi-2 model using LoRA."""
    model, tokenizer = load_model()
    dataset = load_data(DATA_PATH)

    # Tokenization
    def tokenize_function(samples):
        """Tokenizes question-answer pairs for training."""

        # Ensure tokenizer has a padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Convert batched dictionary into formatted Q&A text
        texts = [
            f"Question: {q}\nAnswer: {a}"
            for q, a in zip(samples["prompt"], samples["completion"])
        ]

        # Tokenize inputs
        tokenized_inputs = tokenizer(texts, truncation=True, padding="max_length", max_length=512)

        # Create labels (same as input_ids for causal LM)
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()

        return tokenized_inputs





    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format("torch")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        logging_steps=10,
        save_strategy="epoch",
        push_to_hub=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=training_args
    )

    trainer.train()
    output_dir = "./models/phi2_finetuned_azure"

    # Ensure the directory exists
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Save the fine-tuned model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"[INFO] Model saved successfully to {output_dir}")