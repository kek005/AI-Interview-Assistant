from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Paths
base_model_name = "microsoft/phi-2"  # Ensure this is the correct base model
adapter_dir = "./models/phi2_finetuned_azure"  # Path to fine-tuned LoRA adapter

# Load the base model
model = AutoModelForCausalLM.from_pretrained(base_model_name)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load the fine-tuned adapter
model = PeftModel.from_pretrained(model, adapter_dir)

# Merge LoRA adapter into base model
model = model.merge_and_unload()

# Save the full fine-tuned model
model.save_pretrained("./models/phi2_finetuned_azure_full")
tokenizer.save_pretrained("./models/phi2_finetuned_azure_full")

print("✅ Standalone fine-tuned model saved successfully!")