import os
import openai
import pandas as pd
import time
from tqdm import tqdm

# Load your API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Verify if the API key is set correctly
if openai.api_key is None:
    raise ValueError("OPENAI_API_KEY is not set. Please export it in your environment.")

# Load text from the cleaned .txt file
def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# Split text into manageable chunks (OpenAI’s GPT-4-turbo supports ~4096 tokens)
def split_text(text, chunk_size=2000):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Generate Q&A pairs using GPT-4 API
def generate_qa_pairs(text_chunk):
    prompt = f"""
    You are an AI tutor. Given the following text, generate 3 high-quality question-answer pairs:
    
    TEXT:
    {text_chunk}
    
    OUTPUT FORMAT:
    Q1: [Your Question]
    A1: [Your Answer]
    Q2: [Your Question]
    A2: [Your Answer]
    Q3: [Your Question]
    A3: [Your Answer]
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.7
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error: {e}")
        return None

# Process the text and generate Q&A pairs
def process_text_and_generate_qa(input_txt, output_csv):
    text = load_text(input_txt)
    chunks = split_text(text)

    qa_data = []
    for chunk in tqdm(chunks, desc="Processing Chunks"):
        qa_output = generate_qa_pairs(chunk)
        if qa_output:
            qa_data.append(qa_output)
        time.sleep(1)  # Avoid rate limits

    # Save to CSV
    with open(output_csv, "w", encoding="utf-8") as f:
        f.write("Question,Answer\n")
        for entry in qa_data:
            for line in entry.split("\n"):
                if line.startswith("Q"):
                    question = line.split(": ", 1)[1]
                elif line.startswith("A"):
                    answer = line.split(": ", 1)[1]
                    f.write(f'"{question}","{answer}"\n')

    print(f"✅ Q&A pairs saved to {output_csv}")

# Run the pipeline
input_txt_file = "python_ml_course.txt"  # Your cleaned text file
output_csv_file = "qa_dataset.csv"

process_text_and_generate_qa(input_txt_file, output_csv_file)