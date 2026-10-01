from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

chunks = [
    "program to learn to play checkers, goal is world checkers tournament. performance measure: percent of games it wins in this world tournament",
    "machine learning is field of ai that uses statistical techniques",
    "python is programming language used for data science"
]

for c in chunks:
    emb = model.encode(c)
    print(c[:30], "->", len(emb))