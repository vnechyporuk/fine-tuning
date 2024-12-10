from unsloth import FastLanguageModel
from peft import PeftModel
import torch

max_seq_length = 16384

# Configuration
base_model_name = "unsloth/Phi-3.5-mini-instruct"
lora_adapter_path = "bucket/models/first/checkpoints/checkpoint-2785"  # Path where the LoRA adapter checkpoints were saved
final_model_path = "bucket/models/first/merged/checkpoint-2785"  # Path to save the final merged model
device = "cuda" if torch.cuda.is_available() else "cpu"
#  Reload the base model in half precision
base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_name,
    max_seq_length=max_seq_length,
    dtype=torch.float16,
    load_in_4bit=False  # Reload in higher precision for inference
)
model_to_merge = PeftModel.from_pretrained(base_model.to(device), lora_adapter_path)
merged_model = model_to_merge.merge_and_unload()
if merged_model:
    merged_model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Final merged model saved at: {final_model_path}")
else:
    print("Merging failed.")