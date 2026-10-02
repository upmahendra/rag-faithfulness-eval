from sentence_transformers import SentenceTransformer
import chromadb

# 1. Setup
client = chromadb.Client()
collection = client.get_or_create_collection("day03")
model = SentenceTransformer("all-MiniLM-L6-v2")

docs = [
    "The checkers program learns to play checkers. Performance is percent of games it wins in world tournament.",
    "Machine learning is field that gives computers ability to learn without being explicitly programmed.",
    "Python is a programming language created by Guido van Rossum."
]

# 2. Add embeddings
for i, d in enumerate(docs):
    collection.add(ids=[str(i)], documents=[d], embeddings=[model.encode(d).tolist()])

# 3. Query
query = "what is performance measure for checkers?"
q_emb = model.encode(query).tolist()
result = collection.query(query_embeddings=[q_emb], n_results=1)
context = result['documents'][0][0]

print(f"Context: {context}")

# 4. Simulate two LLM answers
faithful_answer = "Performance measure is percent of games it wins in world tournament."
hallucinated_answer = "Performance measure is accuracy and speed of python code."

# 5. Simple Faithfulness Score = similarity between answer and context
def faithfulness_score(answer, context):
    emb1 = model.encode(answer)
    emb2 = model.encode(context)
    # cosine similarity
    from numpy import dot
    from numpy.linalg import norm
    return dot(emb1, emb2) / (norm(emb1) * norm(emb2))

print(f"\nFaithful Answer: {faithful_answer}")
print(f"Score: {faithfulness_score(faithful_answer, context):.2f} -> FAITHFUL")

print(f"\nHallucinated Answer: {hallucinated_answer}")
print(f"Score: {faithfulness_score(hallucinated_answer, context):.2f} -> NOT FAITHFUL")