import openai
import os
import csv
import time
from split_text.splitter import TextSplitter
from qa_cleaning.cleaner import save_clean_fine_tune_csv  # Import cleaning function
import re


class QuestionGenerator:
    """Class to generate interview-style questions using OpenAI's GPT models."""

    def __init__(self, model="gpt-4o"):
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")  # Read from environment variable
        openai.api_key = self.api_key
        self.splitter = TextSplitter(model=model)  # Initialize the TextSplitter

    def generate_questions(self, input_file, validation_csv, fine_tune_csv):
        """
        Generates Q&A from a text file using OpenAI and saves results to CSV.

        Args:
            input_file (str): Path to the input .txt file.
            validation_csv (str): Path to save the validation CSV.
            fine_tune_csv (str): Path to save the fine-tuning-ready CSV.
        """
        chunks = self.splitter.split_text_from_file(input_file)  # ✅ Correctly split text
        validation_data = []  # Stores (Text Chunk, Generated Question, Generated Answer)
        fine_tune_data = []   # Stores (Question, Answer) only

        for i, chunk in enumerate(chunks):
            print(f"\n--- Chunk {i+1}/{len(chunks)} ---")
            print(chunk)  # PRINT THE CHUNKED DATA BEFORE SENDING TO OPENAI
            prompt = f"""
                Given the following text, generate exactly 3 relevant interview-style questions along with their answers.
                Follow this strict format:

                Question 1: <question text>
                Answer 1: <answer text>

                Question 2: <question text>
                Answer 2: <answer text>

                Question 3: <question text>
                Answer 3: <answer text>

                Do not include any extra characters, numbers, or markdown styling like ** or ###.
                Only return the questions and answers in plain text.

                Text:
                {chunk}
                """

            print(f"\n[INFO] Sending request {i+1}/{len(chunks)} to OpenAI...")  # Log before sending
            try:
                response = self.send_request_with_retry(prompt)
                
                if response:
                    output = response.choices[0].message.content
                    print(f"\n[INFO] OpenAI Raw Response for Chunk {i+1}:\n")
                    print(output)  # ✅ PRINT RAW RESPONSE BEFORE PARSING
                    qa_pairs = self.parse_qa_output(output)  # ✅ Ensure correct parsing

                    # ✅ Keep each chunk aligned with its generated Q&A
                    for question, answer in qa_pairs:
                        validation_data.append((chunk, question, answer))  # Store chunk with each Q&A
                        fine_tune_data.append((question, answer))

                    print(f"[SUCCESS] Received response {i+1}/{len(chunks)} from OpenAI!")  # Log after receiving
                else:
                    print(f"[ERROR] Failed to get response for chunk {i+1}/{len(chunks)}")

            except Exception as e:
                print(f"[ERROR] Exception while processing chunk {i+1}: {e}")

        # ✅ Save results to both CSV files
        self.save_to_csv(validation_csv, validation_data, ["Text Chunk", "Generated Question", "Generated Answer"])
        self.save_to_csv(fine_tune_csv, fine_tune_data, ["Question", "Answer"])

    def send_request_with_retry(self, prompt, max_retries=5, initial_wait=5):
        """
        Sends API request with retry logic in case of rate limits or temporary failures.

        Args:
            prompt (str): The text prompt for OpenAI API.
            max_retries (int): Maximum number of retry attempts.
            initial_wait (int): Initial wait time in seconds before retrying.

        Returns:
            OpenAI response object or None if all retries fail.
        """
        for attempt in range(max_retries):
            try:
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an AI expert generating interview-style questions."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7
                )
                return response  # If request succeeds, return response

            except openai.OpenAIError as e:
                wait_time = initial_wait * (2 ** attempt)  # Exponential backoff
                print(f"[WARNING] OpenAI API error: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

        print("[ERROR] Max retries reached. Skipping this chunk.")
        return None  # Return None if all retries fail

    def parse_qa_output(self, output):
        """Parses the Q&A output into structured (question, answer) pairs."""
        print("\n[DEBUG] Parsing OpenAI Output:\n")
        print(output)  # ✅ PRINT OUTPUT BEFORE PARSING

        qa_pairs = []
        pattern = r"Question \d+: (.*?)\nAnswer \d+: (.*?)(?=\nQuestion|\Z)"  # Regex for structured Q&A

        matches = re.findall(pattern, output, re.DOTALL)

        for question, answer in matches:
            qa_pairs.append((question.strip(), answer.strip()))

        return qa_pairs

    def save_to_csv(self, output_csv, data, headers):
        """Saves data to a CSV file with specified headers."""
        with open(output_csv, mode="w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(data)

# If this file is run directly
if __name__ == "__main__":
    input_file = "data/azure_course.txt"
    validation_csv = "data/output_qa.csv"
    fine_tune_csv = "data/fine_tune_qa.csv"

    generator = QuestionGenerator()
    generator.generate_questions(input_file, validation_csv, fine_tune_csv)

    print(f"\n[INFO] Validation Q&A saved to {validation_csv}")
    print(f"[INFO] Fine-tuning data saved to {fine_tune_csv}")