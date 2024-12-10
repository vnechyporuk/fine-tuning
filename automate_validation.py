from sentence_transformers import SentenceTransformer, util
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download the VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

# Initialize the sentiment analysis and embedding model
sentiment_analyzer = SentimentIntensityAnalyzer()
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Smaller model for quick computation

def get_semantic_similarity(text1, text2):
    # Generate embeddings
    embeddings1 = embedding_model.encode(text1, convert_to_tensor=True)
    embeddings2 = embedding_model.encode(text2, convert_to_tensor=True)
    # Calculate cosine similarity
    semantic_similarity = util.pytorch_cos_sim(embeddings1, embeddings2).item()
    return semantic_similarity

def get_sentiment_score(text):
    # Use VADER to get a compound sentiment score
    sentiment_scores = sentiment_analyzer.polarity_scores(text)
    return sentiment_scores['compound']

def get_sentiment_similarity(text1, text2):
    # Get sentiment scores for each text
    sentiment_score1 = get_sentiment_score(text1)
    sentiment_score2 = get_sentiment_score(text2)
    # Calculate similarity as the inverse of absolute difference
    sentiment_similarity = 1 - abs(sentiment_score1 - sentiment_score2)
    return sentiment_similarity

def sentiment_aware_similarity(text1, text2, weight_semantic=0.5, weight_sentiment=0.5):
    # Get both similarities
    semantic_sim = get_semantic_similarity(text1, text2)
    sentiment_sim = get_sentiment_similarity(text1, text2)
    # Combine them with weights
    total_similarity = (weight_semantic * semantic_sim) + (weight_sentiment * sentiment_sim)
    return total_similarity, semantic_sim, sentiment_sim

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
            #    "storage/val_outputs/first/checkpoint-928.json",
            #    "storage/val_outputs/first/checkpoint-1392.json",
            #    "storage/val_outputs/first/checkpoint-1857.json",
            #    "storage/val_outputs/first/checkpoint-2321.json",
               "storage/val_outputs/first/checkpoint-2785.json",
            #    "storage/val_outputs/first/checkpoint-3249.json",
            #    "storage/val_outputs/first/checkpoint-3714.json",
            #    "storage/val_outputs/first/checkpoint-4178.json",
            #    "storage/val_outputs/first/checkpoint-4640.json",
               ]
import matplotlib.pyplot as plt
plt.switch_backend('pdf')

epochs = list(np.arange(1, len(checkpoints) + 1))
semantic_scores_per_epoch = []
sentiment_score_per_epoch = []
for checkpoint in checkpoints:
    dataset = load_dataset('vibhorag101/phr-mental-therapy-dataset-conversational-format-1024-tokens', split = "val")
    print(f"\nCHECKPOINT - {checkpoint}\n")

    with open(checkpoint, 'r') as file:
        generated_prompts = json.load(file)  # Load JSON data from file

    n = len(generated_prompts.keys())

    global_semantic_sim = 0
    global_sentiment_sim = 0
    for i, index in enumerate(generated_prompts.keys()):
        print(f"index = {index}, {i + 1}th chat from {n} chats")
        messages = dataset[int(index)]['messages']
        prompt = ''
        chat_semantic_sim = 0
        chat_sentiment_sim = 0
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
                total_similarity, semantic_sim, sentiment_sim  = sentiment_aware_similarity(original_response, bot_response)
                print(f"semantic similarity: {semantic_sim}")
                print(f"sentiment similarity: {sentiment_sim}")
                chat_semantic_sim += semantic_sim
                chat_sentiment_sim += sentiment_sim
                prompt += f'<|assistant|> {original_response}<|end|>'
                n_of_bot_messages += 1
        chat_semantic_sim /= n_of_bot_messages
        chat_sentiment_sim /= n_of_bot_messages
        print(f"chat semantic score = {chat_semantic_sim}\n")
        print(f"chat sentiment score = {chat_sentiment_sim}\n")
        global_semantic_sim += chat_semantic_sim
        global_sentiment_sim += chat_sentiment_sim
    global_semantic_sim /= n
    global_sentiment_sim /= n
    print(f"Epoch semantic score = {global_semantic_sim}")
    print(f"Epoch sentiment score = {global_sentiment_sim}")
    semantic_scores_per_epoch.append(global_semantic_sim)
    sentiment_score_per_epoch.append(global_sentiment_sim)



plt.title("Semantic score per epoch")
plt.xlabel("Epoch")
plt.ylabel("Semantic score")
plt.plot(epochs, semantic_scores_per_epoch)
plt.savefig("semantic_score_per_epoch.png")
plt.clf()

plt.title("Sentiment score per epoch")
plt.xlabel("Epoch")
plt.ylabel("Sentiment score")
plt.plot(epochs, sentiment_score_per_epoch)
plt.savefig("sentiment_score_per_epoch.png")
                