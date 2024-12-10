from llama_cpp import Llama
max_seq_length = int(16384 / 8)

def load(path):
  global llm
  llm = Llama(
    model_path=path,  # path to GGUF file
    n_ctx=max_seq_length,  # The max sequence length to use - note that longer sequence lengths require much more resources
    n_threads=4, # The number of CPU threads to use, tailor to your system and the resulting performance
    n_gpu_layers=35, # The number of layers to offload to GPU, if you have GPU acceleration available. Set to 0 if no GPU acceleration is available on your system.
  )


# Encode the prompt text and generate tokens
def phi_3_generate(chat_history):
    prompt = ""
    for from_, text in chat_history:
        prompt += f"<|{from_}|>\n {text}\n<|end|>\n"
    prompt += "<|assistant|>\n"
    print(prompt)
    output = llm(
      prompt,
      max_tokens=max_seq_length,
      stop=["<|end|>"],
      echo=False,  # Whether to echo the prompt
    )

    output = output['choices'][0]['text']
    return output