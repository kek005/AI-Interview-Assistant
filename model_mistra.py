import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ✅ Load a QUANTIZED model (4-bit GPTQ) for FAST inference
model_name = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"

# 🔥 Load Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # ✅ Auto-load on best GPU
    torch_dtype=torch.float16,  # ✅ FP16 for max speed
    trust_remote_code=True
)

# 🔥 Compile model for speed boost (optional but recommended)
model = torch.compile(model)

print("✅ Model Loaded in VRAM & Ready for Ultra-Fast Inference!")

# 🚀 Function to generate AI response with maximum determinism
def generate_response(prompt):
    start_time = time.perf_counter()

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to("cuda")

    # ✅ Generate output tokens with **fully deterministic settings**
    output_tokens = model.generate(
        **inputs,
        max_new_tokens=30,  # 🔥 Limit new tokens only (Faster)
        num_return_sequences=1,  # ✅ Generate only 1 response
        do_sample=False,  # ✅ **Greedy decoding (Deterministic)**
        temperature=0.0,  # ✅ **Fully deterministic**
        top_k=None,  # ✅ **Considers all possible tokens**
    )

    response_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    end_time = time.perf_counter()
    print(f"✅ Response generated in {(end_time - start_time) * 1000:.2f} ms")

    # ✅ Print response instantly
    print("\n🤖 AI Response:\n", response_text, "\n")

    return response_text

# 🏆 Keep model in VRAM & ready for real-time inference
print("🚀 Ready! Enter your prompt below:")
while True:
    prompt = input("\n💬 Enter Prompt: ")
    if prompt.lower() in ["exit", "quit"]:
        print("👋 Exiting...")
        break
    generate_response(prompt)  # ✅ Now fully deterministic!