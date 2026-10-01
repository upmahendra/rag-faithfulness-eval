import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.Client()
collection = client.create_collection("book")

chunks = [
    "program to learn to play checkers, goal is world checkers tournament. performance measure: percent of games it wins in this world tournament",
    "machine learning is field of ai that uses statistical techniques"
]

for i, c in enumerate(chunks):
    collection.add(ids=[str(i)], embeddings=[model.encode(c).tolist()], documents=[c])

query = "what is performance measure for checkers?"
results = collection.query(query_embeddings=[model.encode(query).tolist()], n_results=1)

print(results['documents'][0][0])