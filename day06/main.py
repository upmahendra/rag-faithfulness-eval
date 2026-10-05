from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.Client(Settings())
collection = client.get_or_create_collection("day06")

# Long doc example
long_doc = """
The checkers program learns to play checkers.
The task T is playing checkers.
The performance measure P is percent of games won.
The experience E is playing against itself.
Machine learning is about improving P with E.
Arthur Samuel built first checkers program in 1959.
It used minimax search and learned from self-play.
Performance evaluation is crucial in ML.
"""

def chunk_text(text, chunk_size=20, overlap=5):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

# Strategy 1: No overlap
chunks_no_overlap = chunk_text(long_doc, chunk_size=20, overlap=0)
print(f"No Overlap chunks: {len(chunks_no_overlap)}")
print(chunks_no_overlap[0])

# Strategy 2: With overlap
chunks_overlap = chunk_text(long_doc, chunk_size=20, overlap=5)
print(f"\nOverlap chunks: {len(chunks_overlap)}")
print(chunks_overlap[0])
print(chunks_overlap[1])

# Add overlap chunks to ChromaDB
for idx, chunk in enumerate(chunks_overlap):
    collection.add(
        documents=[chunk],
        ids=[f"chunk_{idx}"],
        embeddings=[model.encode(chunk).tolist()]
    )

# Test retrieval
query = "what is performance measure for checkers"
q_emb = model.encode(query).tolist()

results = collection.query(query_embeddings=[q_emb], n_results=2)
print(f"\nQuery: {query}")
print(f"Top contexts: {results['documents'][0]}")

print("\nDay 06 Done - Chunking implemented")