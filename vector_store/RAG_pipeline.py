import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load vector store
index = faiss.read_index("vector_store/faiss_index.index")

with open("vector_store/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

with open("vector_store/documents.pkl", "rb") as f:
    documents = pickle.load(f)

# Models
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
generator = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.2",
    max_new_tokens=300
)

def retrieve(query, k=5):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)

    results = []
    for i in indices[0]:
        results.append({
            "text": documents[i],
            "metadata": metadata[i]
        })
    return results

def generate_answer(query):
    retrieved = retrieve(query)

    context = "\n\n".join([r["text"] for r in retrieved])

    prompt = f"""
You are a financial analyst assistant for CrediTrust.
Use ONLY the context below to answer the question.
If the answer is not contained in the context, say you do not have enough information.

Context:
{context}

Question:
{query}

Answer:
"""

    response = generator(prompt)[0]["generated_text"]
    return response, retrieved
