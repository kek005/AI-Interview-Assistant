from transformers import pipeline
from config import OUTPUT_DIR

def evaluate():
    """Loads and tests the fine-tuned model."""
    model_path = OUTPUT_DIR
    model, tokenizer = load_model()
    qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Example test
    question = "What is Microsoft Entra ID?"
    result = qa_pipeline(question, max_length=200)
    print(result[0]['generated_text'])