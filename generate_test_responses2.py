import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import numpy as np
from datasets import load_dataset
import json
max_seq_length = int(16384 / 8) # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
checkpoints = [
    # "storage/models/first/checkpoints/checkpoint-464",
    #            "storage/models/first/checkpoints/checkpoint-928",
    #            "storage/models/first/checkpoints/checkpoint-1392",
    #            "storage/models/first/checkpoints/checkpoint-1857",
    #            "storage/models/first/checkpoints/checkpoint-2321",
               "storage/models/first/checkpoints/checkpoint-2785",
    #            "storage/models/first/checkpoints/checkpoint-3249",
    #            "storage/models/first/checkpoints/checkpoint-3714",
    #            "storage/models/first/checkpoints/checkpoint-4178",
            #    "storage/models/first/checkpoints/checkpoint-4640",
               ]
dataset = load_dataset("vibhorag101/phr-mental-therapy-dataset-conversational-format-1024-tokens", split = "test")
n = 10
arr = np.arange(n)  # Generate array from 0 to n-1
for checkpoint_path in checkpoints:
    results = {}
    model, tokenizer = FastLanguageModel.from_pretrained(
        checkpoint_path,
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

    # Encode the prompt text and generate tokens
    def model_generate(prompt):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Generate text
        with torch.no_grad():
            output = model.generate(input_ids, max_length=2000, num_return_sequences=1)

        # Decode the generated text
        generated_text = tokenizer.decode(output[0], skip_special_tokens=False)
        return generated_text


    print(f"checkpoint - {checkpoint_path}")
    indexes = arr[0:n]
    for i, index in enumerate(indexes):
        print(f"index = {index}, {i + 1}th chat from {n} chats")
        results[int(index)] = []
        messages = dataset[int(index)]['messages']
        prompt = ''
        n_of_bot_messages = 0
        for message in messages:
            if message['role'] == 'system':
                continue
            content = message['content']
            if message['role'] == 'user':
                prompt += f'<|user|>\n{content}<|end|>\n'
            if message['role'] == 'assistant':
                original_response = content
                genereted_text = model_generate(prompt + "<|assistant|>\n")
                prompt += f'<|assistant|>\n{original_response}<|end|>\n'
                n_of_bot_messages += 1
                results[int(index)].append(genereted_text)
    print(results)
    save_to = checkpoint_path.replace("storage/models/first/checkpoints/", "")
    with open(f"bucket/test_outputs/first/{save_to}.json", "w") as file:
         json.dump(results, file, indent=4)
    print(f"saved to bucket/test_outputs/first/{save_to}.json")
print(results)