from transformers import pipeline

def generate_qa_pairs(text_file, output_csv):
    """Generates question-answer pairs from text and saves them to a CSV file."""
    import csv

    # Load the question generation pipeline (T5 model)
    qa_pipeline = pipeline("text2text-generation", model="t5-small")

    # Read cleaned text
    with open(text_file, "r", encoding="utf-8") as file:
        text = file.read()

    # Split the text into sentences for better question generation
    sentences = text.split(". ")  # You can improve this with NLP sentence segmentation
    
    qa_pairs = []
    
    for sentence in sentences:
        input_text = f"generate question: {sentence}"
        
        # Generate question using T5 model
        question = qa_pipeline(input_text)[0]['generated_text']
        
        # Store the (question, answer) pair
        qa_pairs.append([question, sentence])

    # Save to CSV file
    with open(output_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Question", "Answer"])
        writer.writerows(qa_pairs)

    print(f"Generated Q&A saved to {output_csv}")

# Example Usage
cleaned_text_file = "/home/keke/python_ml_course_cleaned.txt"  # Use cleaned text
output_csv_file = "/home/keke/python_ml_course_QA.csv"

generate_qa_pairs(cleaned_text_file, output_csv_file)