import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import numpy as np
from datasets import load_dataset
import json
max_seq_length = int(16384 / 8) # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
model, tokenizer = FastLanguageModel.from_pretrained(
        "bucket/models/first/merged/checkpoint-2785",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

model.eval()

    # Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "phi-3", # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
    mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
)
FastLanguageModel.for_inference(model)

model.save_pretrained_gguf("storage/models/first/exported/epoch6", tokenizer, quantization_method = "f16")