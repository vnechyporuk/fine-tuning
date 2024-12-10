from sentence_transformers import SentenceTransformer, util

sentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2')

def check_semantic_similarity(text1, text2):

    # Encode sentences into vectors
    embeddings1 = sentenceTransformer.encode(text1, convert_to_tensor=True)
    embeddings2 = sentenceTransformer.encode(text2, convert_to_tensor=True)

    # Compute cosine similarity
    cosine_score = util.pytorch_cos_sim(embeddings1, embeddings2).item()

    # Determine if they are similar based on the threshold
    return cosine_score

import torch
import numpy as np
from datasets import load_dataset
max_seq_length = int(16384 / 8) # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

dataset = load_dataset("vibhorag101/phr-mental-therapy-dataset-conversational-format-1024-tokens", split = "val")
n = len(dataset)
arr = np.arange(n)  # Generate array from 0 to n-1
#np.random.shuffle(arr)  # Shuffle the array

#n = int(input("How many samples use to evaluate: "))
indexes = arr[0:n]
embeddings = []
for i, index in enumerate(indexes):
    print(f"index = {index}, {i + 1}th chat from {n} chats")
    messages = dataset[int(index)]['messages']
    prompt = ''
    n_of_bot_messages = 0
    for message in messages:
        if message['role'] == 'system':
            continue
        content = message['content']
        if message['role'] == 'user':
            prompt += f'<user>\n{content}<end>\n'
        if message['role'] == 'assistant':
            original_response = content
            prompt += original_response
    embeddings.append(sentenceTransformer.encode(prompt, convert_to_numpy=True))
embeddings = np.array(embeddings)

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
plt.switch_backend('pdf')

scores = []
q = []
for n_clusters in range(2, 100):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
    k_means_labels = kmeans.fit_predict(embeddings)
    kmeans_silhouette = silhouette_score(embeddings, k_means_labels)
    scores.append(kmeans_silhouette)
    q.append(n_clusters)
plt.plot(q, scores)
plt.title("silhouette score per number of clusters")
plt.xlabel("num of clusters")
plt.ylabel("silhouette score")
plt.savefig("sentence_clusters_score_plot.png")