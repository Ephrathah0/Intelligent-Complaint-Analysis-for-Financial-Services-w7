import pandas as pd
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import pickle

# 1. Load cleaned complaints
filtered_df = pd.read_csv("data/processed/filtered_complaints.csv")

# 2. Sampling (Task 2)
TARGET_PRODUCTS = ["Credit card", "Personal loan", "Savings account", "Money transfer"]
sampled_df = (
    filtered_df.groupby("Product", group_keys=False)
    .apply(lambda x: x.sample(frac=0.2, random_state=42))
)

# 3. Chunking
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
documents = []
for _, row in sampled_df.iterrows():
    chunks = splitter.split_text(row["cleaned_narrative"])
    for chunk in chunks:
        documents.append({
            "text": chunk,
            "metadata": {"complaint_id": row["Complaint ID"], "product": row["Product"]}
        })

# 4. Embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")
texts = [d["text"] for d in documents]
embeddings = embedder.encode(texts, show_progress_bar=True)

# 5. FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))
faiss.write_index(index, "vector_store/faiss.index")

# 6. Save metadata
with open("vector_store/metadata.pkl", "wb") as f:
    pickle.dump(documents, f)


