import numpy as np
from datasets import load_dataset
import json

# Encode the prompt text and generate tokens
def chatbot_response(prompt, generated_prompt):

    # Decode the generated text
    generated_text = generated_prompt
    generated_text = generated_text.replace(prompt, "")
    while generated_text[0] == ' ' or generated_text[0] == '\n':
        generated_text = generated_text[1:]
    if generated_text[0] == '<':
        while generated_text[0] != '>':
            generated_text = generated_text[1:]
        generated_text = generated_text[1:]
    for i in range(len(generated_text)):
        if generated_text[i] == '<':
            generated_text = generated_text[0:i]
            break
    return generated_text

checkpoints = [
    # "storage/val_outputs/first/checkpoint-464.json",
    #            "storage/val_outputs/first/checkpoint-928.json",
    #            "storage/val_outputs/first/checkpoint-1392.json",
    #            "storage/val_outputs/first/checkpoint-1857.json",
    #            "storage/val_outputs/first/checkpoint-2321.json",
               "storage/val_outputs/first/checkpoint-2785.json",
            #    "storage/val_outputs/first/checkpoint-3249.json",
            #    "storage/val_outputs/first/checkpoint-3714.json",
            #    "storage/val_outputs/first/checkpoint-4178.json",
            #    "storage/val_outputs/first/checkpoint-4640.json",
               ]
import matplotlib.pyplot as plt
plt.switch_backend('pdf')

epochs = list(np.arange(1, len(checkpoints) + 1))
for checkpoint in checkpoints:
    dataset = load_dataset('vibhorag101/phr-mental-therapy-dataset-conversational-format-1024-tokens', split = "val")
    print(f"\nCHECKPOINT - {checkpoint}\n")

    with open(checkpoint, 'r') as file:
        generated_prompts = json.load(file)  # Load JSON data from file

    n = len(generated_prompts.keys())
    
    epoch_score = 0
    for i, index in enumerate(generated_prompts.keys()):
        print(f"index = {index}, {i + 1}th chat from {n} chats")
        messages = dataset[int(index)]['messages']
        prompt = ''
        chat_score = 0
        n_of_bot_messages = 0
        for message in messages:
            if message['role'] == 'system':
                continue
            content = message['content']
            if message['role'] == 'user':
                print(f"user: {content}\n")
                prompt += f'<|user|> {content}<|end|>'
            if message['role'] == 'assistant':
                generated_prompt = generated_prompts[index][n_of_bot_messages]
                original_response = content
                bot_response = chatbot_response(prompt, generated_prompt)
                print(f"original response:\n{original_response}\nbotresponse:\n{bot_response}")
                chat_score += float(input("Give a score for a message: "))
                prompt += f'<|assistant|> {original_response}<|end|>'
                n_of_bot_messages += 1
        chat_score /= n_of_bot_messages
        print(f"chat score = {chat_score}\n")
        epoch_score += chat_score
    epoch_score /= n
    print(f"Epoch score = {epoch_score}")
