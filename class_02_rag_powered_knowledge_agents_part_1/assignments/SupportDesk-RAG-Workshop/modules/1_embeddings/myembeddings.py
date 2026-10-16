import os

from dotenv import load_dotenv 
from openai import OpenAI       # OpenAI API client for generating embeddings
import json
import numpy as np              # For numerical operations on embedding vectors
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity  # Measure similarity between vectors

print("\nLoading support tickets...")

file = Path(__file__).resolve().parents[2] / "data/synthetic_tickets.json"
with open(file, 'r') as f:
    tickets = json.load(f)

print(type(tickets))


ticket_texts = [
    str(ticket) 
    for ticket in tickets
]

def myembedding ():
    
    # Load environment variables
    root_env = Path(__file__).resolve().parents[5] / ".env"
    load_dotenv(root_env)

    embedding_model = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
    embedding_dim = 1536  # Number of dimensions in the embedding vector

    print("Initializing OpenAI client...")
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    # Generate embedding for all  tickets
    response = client.embeddings.create(input=ticket_texts, model=embedding_model)
    embeddings = np.array([data.embedding for data in response.data])

    # Generate embedding for the query  
    query = "Users can't login after changing password"
    query_response = client.embeddings.create(input=[query], model=embedding_model)
    query_embedding = np.array([query_response.data[0].embedding])

    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_k = 5
    top_indices = np.argsort(similarities)[::-1][:top_k]
    print(top_indices)

    for rank, idx in enumerate(top_indices, 1):
        ticket = tickets[idx]
        score = similarities[idx]
        print(f"\n#{rank} - Similarity: {score:.4f}")
        print(f"Ticket ID: {ticket['ticket_id']}")
        print(f"Title: {ticket['title']}")
        print(f"Category: {ticket['category']} | Priority: {ticket['priority']}")
        print(f"Description: {ticket['description'][:150]}...")
myembedding()