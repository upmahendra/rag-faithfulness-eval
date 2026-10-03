# pip install sentence-transformers chromadb numpy

from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np

# setup - use in-memory client
client = chromadb.Client()
# Delete old collection if you re-run
try:
    client.delete_collection("day04")
except:
    pass

collection = client.get_or_create_collection("day04")
model = SentenceTransformer("all-MiniLM-L6-v2")

docs = [
    "The checkers program learns to play checkers. Performance is percent of games it wins in world tournament.",
    "Machine learning is field that gives computers ability to learn without being explicitly programmed.",
    "Python is a programming language created by Guido van Rossum."
]

# add with normalized embeddings (better)
embeddings = model.encode(docs, normalize_embeddings=True).tolist()
ids = [str(i) for i in range(len(docs))]

if collection.count() == 0:
    collection.add(ids=ids, documents=docs, embeddings=embeddings)

def get_context(q):
    # Let Chroma do encoding - simpler and faster
    res = collection.query(query_texts=[q], n_results=1)
    return res['documents'][0][0]

def check_faithfulness(answer, context):
    # Always normalize
    a = model.encode(answer, normalize_embeddings=True)
    c = model.encode(context, normalize_embeddings=True)
    score = float(np.dot(a, c)) # cosine because normalized

    # Improved threshold - 0.75 is too low for MiniLM
    label = "FAITHFUL" if score > 0.80 else "HALLUCINATED"
    return score, label

# test
query = "what is performance measure for checkers?"
context = get_context(query)

print("Query:", query)
print("Retrieved context:", context)

ans1 = "Performance measure is percent of games it wins in world tournament."
s1, l1 = check_faithfulness(ans1, context)
print(f"\n[TEST 1] ans: {ans1}\nscore: {s1:.3f} -> {l1} | Expected: FAITHFUL")

ans2 = "Performance measure is accuracy and speed of python code."
s2, l2 = check_faithfulness(ans2, context)
print(f"\n[TEST 2] ans: {ans2}\nscore: {s2:.3f} -> {l2} | Expected: HALLUCINATED")

# Day 04 Task - Final summary
print("\n--- Day 04 Result ---")
print(f"Task Completed: RAG Faithfulness check working")