import requests
import json
import time
import torch

# Define the request payload
payload = {
    "model": "llama3",
    "prompt": "What is software testing?",
    "stream": False,  # Keep it false for fast inference
    "max_tokens": 30,  # Limits response length
    "temperature": 0.5,  # Slight randomness to avoid repetition
    "top_k": 50,  # Limits vocab choices for diversity
    "top_p": 0.9  # Nucleus sampling
}

OLLAMA_URL = "http://localhost:11434/api/generate"

# Measure inference time
start_time = time.time()

# Send request to Ollama's API
response = requests.post(OLLAMA_URL, json=payload)

# Measure end time
end_time = time.time()
inference_time = end_time - start_time

# Parse and print the response
result = response.json()
print("\nOllama Response:", result["response"])
print(f"⏱️ Inference Time: {inference_time:.3f} seconds")

# Check if GPU is being used
if torch.cuda.is_available():
    print(f"✅ GPU is available: {torch.cuda.get_device_name(0)}")
    print(f"🔥 GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e6} MB")
else:
    print("⚠️ GPU is NOT being used. Running on CPU.")