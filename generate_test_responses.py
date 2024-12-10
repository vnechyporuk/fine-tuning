from deploy.fine_tuned_phi_3_chat import phi_3_generate as generate_response, load
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
    if checkpoint_path == "storage/models/first/checkpoints/checkpoint-2785":
        load("./bucket/models/first/exported/epoch6/unsloth.F16.gguf")
    if checkpoint_path == "storage/models/first/checkpoints/checkpoint-464":
        load("./bucket/models/first/exported/epoch1/unsloth.F16.gguf")
    results = {}

    print(f"checkpoint - {checkpoint_path}")
    indexes = arr[0:n]
    for i, index in enumerate(indexes):
        print(f"index = {index}, {i + 1}th chat from {n} chats")
        results[int(index)] = []
        messages = dataset[int(index)]['messages']
        prompt = []
        n_of_bot_messages = 0
        for message in messages:
            if message['role'] == 'system':
                continue
            content = message['content']
            if message['role'] == 'user':
                prompt.append((message['role'], content))
            if message['role'] == 'assistant':
                original_response = content
                genereted_text = generate_response(prompt)
                print(f"BOT RESPONSE: {genereted_text}\n")
                prompt.append((message['role'], content))
                n_of_bot_messages += 1
                results[int(index)].append(genereted_text)
        print(results)
    save_to = checkpoint_path.replace("storage/models/first/checkpoints/", "")
    with open(f"bucket/test_outputs/first/{save_to}.json", "w") as file:
         json.dump(results, file, indent=4)
    print(f"saved to bucket/test_outputs/first/{save_to}_second.json")
print(results)