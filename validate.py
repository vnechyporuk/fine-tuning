from llama_cpp import Llama


llm = Llama(
  model_path="storage/models/first/unsloth.Q4_K_M.gguf",  # path to GGUF file
  n_ctx=int(16384/2),  # The max sequence length to use - note that longer sequence lengths require much more resources
  n_threads=1, # The number of CPU threads to use, tailor to your system and the resulting performance
  n_gpu_layers=8, # The number of layers to offload to GPU, if you have GPU acceleration available. Set to 0 if no GPU acceleration is available on your system.
)


def chatbot_response(chat_history):
    prompt = ""
    for i, message in enumerate(chat_history):
        if i % 2 == 0:
            prompt += f"<|user|>\n{message}<|end|>\n"
        else:
            prompt += f"<|assistant|>\n{message}<|end|>\n"
    prompt += '<|assistant|>'

    output = llm(
      prompt,
      max_tokens=256,  # Generate up to 256 tokens
      stop=["<|end|>"],
      echo=True,  # Whether to echo the prompt
    )
    output = output['choices'][0]['text'][len(prompt):]
    return output


from datasets import load_dataset
import numpy as np

dataset = load_dataset("vibhorag101/phr-mental-therapy-dataset-conversational-format-1024-tokens", split = "val")
n = len(dataset)
arr = np.arange(n)  # Generate array from 0 to n-1
np.random.shuffle(arr)  # Shuffle the array

n = int(input("How many samples use to evaluate: "))
print(dataset.list_indexes)
indexes = arr[0:n]
global_score = 0
for index in indexes:
    print(f"index = {index}")
    messages = dataset[int(index)]['messages']
    chat_history = []
    chat_score = 0
    n_of_bot_messages = 0
    for message in messages:
        if message['role'] == 'system':
            continue
        if message['role'] == 'user':
            chat_history.append(message['content'])
        if message['role'] == 'assistant':
            for i, chat_message in enumerate(chat_history):
                if i % 2 == 0:
                    print(f"human:\n{chat_message}\n")
                else:
                    print(f"bot:\n{chat_message}\n")
            original_response = message['content']
            bot_response = chatbot_response(chat_history)
            print(f"original response:\n{original_response}\nbotresponse:\n{bot_response}")
            score = float(input("Evaluate bot response (0.0 to 1.0): "))
            chat_score += score
            chat_history.append(original_response)
            n_of_bot_messages += 1
    chat_score /= n_of_bot_messages
    print(f"chat score = {chat_score}")
    global_score += chat_score
global_score /= n
print(f"Global score = {global_score}")
            
        
            
