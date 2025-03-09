import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model into memory (ONLY GPU, no CPU fallback)
model_name = "microsoft/phi-2"

if not torch.cuda.is_available():
    raise RuntimeError("❌ No GPU detected! This script requires a CUDA-compatible GPU.")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)

print("✅ Phi-2 Model Loaded on GPU. Start chatting! (Type 'exit' to quit)")

# Continuous interactive loop
while True:
    user_input = input("\nYou: ").strip()
    if user_input.lower() == "exit":
        print("🚪 Exiting...")
        break

    if not user_input:
        print("⚠️ Please enter a valid question.")
        continue

    # Tokenize input
    inputs = tokenizer(user_input, return_tensors="pt").to("cuda")

    if inputs["input_ids"].size(1) == 0:
        print("⚠️ Error: Empty input detected. Please try again.")
        continue  # Prevents crashes

    # Start timing inference
    start_time = time.time()

    # Generate response (optimized settings)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=150,  # Limits response length
            do_sample=False,  # Disables randomness for accuracy
            repetition_penalty=2.2,  # Stronger penalty for repeats
            encoder_no_repeat_ngram_size=3,  # Prevents redundant phrases
            top_k=10,  # Reduces unexpected words
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    # End timing
    end_time = time.time()
    inference_time = end_time - start_time

    # Decode and print response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print(f"🧠 Phi-2: {response}")
    print(f"⏱️ Response Time: {inference_time:.3f} seconds")