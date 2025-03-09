import re
import csv

def clean_text(text):
    """
    Cleans the generated text by removing unnecessary special characters
    and standardizing formatting.
    """
    # Remove Markdown-style bold (**text**) and numbering (e.g., "Question 1:")
    text = re.sub(r"\*\*|\bQuestion \d+:?", "", text).strip()
    
    return text

def save_clean_fine_tune_csv(output_csv, qa_pairs):
    """
    Saves cleaned Q&A pairs to a CSV file formatted for fine-tuning.

    Args:
        output_csv (str): Path to the output CSV file.
        qa_pairs (list of tuples): List of (question, answer) pairs.
    """
    with open(output_csv, mode="w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Question", "Answer"])  # Fine-tune column headers

        for question, answer in qa_pairs:
            clean_q = clean_text(question)
            clean_a = clean_text(answer)
            writer.writerow([clean_q, clean_a])

    print(f"[INFO] Fine-tune data saved to: {output_csv}")