import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 🚨 Fail if no GPU is available
if not torch.cuda.is_available():
    raise RuntimeError("🚨 No GPU detected! This model requires CUDA for real-time performance.")

# Set model path
MODEL_PATH = "./models/phi2_finetuned_azure_full"

# Force GPU execution
device = torch.device("cuda")

# 🔥 Enable 4-bit Quantization for Max Speed
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_compute_dtype=torch.float16  # ✅ Prevents dtype mismatch warning
)

# 🔥 Load Tokenizer with Fast Mode
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)

# 🔥 Load Model with Maximum Optimizations
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    torch_dtype=torch.float16,  # Switch to FP16 for extra speed
    attn_implementation="flash_attention_2",  # Optimized for RTX 40-series
    device_map="auto"
)

# 🔥 Compile Model for Even Faster Execution
model = torch.compile(model)

# ✅ Set pad_token_id to Avoid Warnings & Speed Up Execution
model.config.pad_token_id = tokenizer.eos_token_id

# ✅ Warm-up Inference (50 Trials)
print("\n🔥 Warming up model... (50 trials, results hidden)")
warmup_trials = 50
times = []
dummy_input = tokenizer("Question: What is Microsoft Entra ID?\nAnswer:", return_tensors="pt").to(device)

for _ in range(warmup_trials):
    start_time = time.time()
    _ = model.generate(**dummy_input, max_length=30, pad_token_id=tokenizer.eos_token_id, use_cache=True)
    print("warming up")
    end_time = time.time()
    times.append(end_time - start_time)

print(f"✅ Warm-up complete. Avg speed: {sum(times) / warmup_trials:.4f} sec\n")

# 🤖 AI Interview Assistant Interactive Mode
print("🤖 AI Interview Assistant is running... Type your question and press Enter.")
print("🔴 Type 'exit' to stop.\n")

while True:
    question = input("❓ Enter your question: ")
    if question.lower() == "exit":
        print("🛑 Exiting AI Interview Assistant. Goodbye!")
        break
    
    # ✅ Prepare Input (Fixed `.to(device)`)
    inputs = tokenizer(f"Question: {question}\nAnswer:", return_tensors="pt").to(device)

    # ⏱️ Measure Response Time
    start_time = time.time()
    outputs = model.generate(**inputs, max_length=70, pad_token_id=tokenizer.eos_token_id, use_cache=True)
    end_time = time.time()

    # ✅ Print Response
    print("\n🤖 Model Response:", tokenizer.decode(outputs[0], skip_special_tokens=True))
    print(f"⚡ Response Time: {end_time - start_time:.4f} sec\n")