import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# 🚨 Fail if no GPU is available
if not torch.cuda.is_available():
    raise RuntimeError("🚨 No GPU detected! This model requires CUDA for real-time performance.")

# ✅ Set model path
MODEL_PATH = "TheBloke/OpenHermes-2.5-Mistral-7B-AWQ"

# ✅ Print GPU Information
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Count: {torch.cuda.device_count()}")
print(f"Current GPU: {torch.cuda.current_device()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# ✅ Force GPU Execution
device = torch.device("cuda:0")

# ✅ Enable TensorFloat-32 for Faster Computation
torch.backends.cuda.matmul.allow_tf32 = True

# ✅ Load Tokenizer (Fast Mode)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)

# ✅ Load Model with Maximum VRAM Utilization
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,  # Use FP16 for high speed
    attn_implementation="flash_attention_2",  # Optimized attention mechanism
    device_map={"": 0},  # 🚀 Forces full GPU execution (instead of auto)
    use_cache=True  # Enables KV caching for speed boost
)

# ✅ Compile Model for Faster Execution
model = torch.compile(model)

# ✅ Set pad_token_id to Avoid Warnings & Speed Up Execution
model.config.pad_token_id = tokenizer.eos_token_id

# ✅ Increase KV Cache Size to Optimize VRAM Usage
model.config.use_cache = True  # Ensure KV cache is enabled
model.config.kv_cache_size = 4096  # 🚀 Increase cache for fast decoding

# ✅ Warm-up Inference (50 Trials)
print("\n🔥 Warming up model... (50 trials, results hidden)")
warmup_trials = 50
times = []
dummy_input = tokenizer("Question: What is Microsoft Entra ID?\nAnswer:", return_tensors="pt").to(device)

for _ in range(warmup_trials):
    start_time = time.time()
    _ = model.generate(**dummy_input, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
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
    
    # ✅ Prepare Input
    inputs = tokenizer(f"Question: {question}\nAnswer:", return_tensors="pt").to(device)

    # ✅ Dynamically Adjust Max Tokens for Concise Answers
    max_new_tokens = min(len(question.split()) * 2, 100)  # Adjust dynamically

    # ⏱️ Measure Response Time
    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,  # Optimized token length
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=1,
        use_cache=True  # Reuse KV cache for faster decoding
    )
    end_time = time.time()

    # ✅ Process & Print Response (Remove Question Repetition)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    if response.startswith(question):
        response = response[len(question):].strip()  # Remove redundant question

    print("\n🤖 Model Response:", response)
    print(f"⚡ Response Time: {end_time - start_time:.4f} sec\n")