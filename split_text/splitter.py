import tiktoken
import nltk

import os

nltk.download("punkt")  # Ensure sentence tokenizer is available

class TextSplitter:
    """Class to handle text splitting into token-based chunks while maintaining sentence integrity."""

    def __init__(self, model="gpt-4o", max_tokens=512):
        self.model = model
        self.max_tokens = max_tokens
        self.encoding = tiktoken.encoding_for_model(model)

    def split_text_from_file(self, file_path):
        """
        Reads text from a file and splits it into chunks of max_tokens while keeping sentences intact.

        Args:
            file_path (str): Path to the input .txt file.

        Returns:
            List[str]: List of text chunks.
        """
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            return []

        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read().strip()  # Read and clean text

        sentences = nltk.sent_tokenize(text)  # Tokenize into sentences
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_tokens = len(self.encoding.encode(sentence))  # Count tokens in sentence

            # If adding the next sentence exceeds max_tokens, store the chunk and start a new one
            if current_length + sentence_tokens > self.max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

            current_chunk.append(sentence)
            current_length += sentence_tokens

        # Add the last chunk if it has content
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

# If this file is run directly, allow it to take input from the command line
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python splitter.py <input_file_path>")
    else:
        input_path = sys.argv[1]
        splitter = TextSplitter()
        chunks = splitter.split_text_from_file(input_path)
        for i, chunk in enumerate(chunks):
            print(f"\n--- Chunk {i+1} ({len(splitter.encoding.encode(chunk))} tokens) ---\n")
            print(chunk)