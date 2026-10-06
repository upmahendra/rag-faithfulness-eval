from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.Client(Settings(anonymized_telemetry=False))
collection = client.get_or_create_collection("day07")

pdf_text = """
Machine learning is a field of AI.

The checkers program learns to play checkers. The task T is playing checkers.
The performance measure P is percent of games won in the world tournament.
The experience E is obtained by playing games against itself.

Arthur Samuel was a pioneer in machine learning. He built the first checkers program in 1959.
The program used minimax search algorithm. It improved through self-play.
Performance evaluation is done by win rate against opponents.
"""

def recursive_chunk(text, chunk_size=150, overlap=20):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) <= chunk_size:
            current_chunk += " " + para if current_chunk else para
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
                # overlap implementation
                current_chunk = current_chunk[-overlap:] if len(current_chunk) > overlap else ""

            if len(para) > chunk_size:
                sentences = para.split(". ")
                temp = ""
                for sent in sentences:
                    if len(temp) + len(sent) <= chunk_size:
                        temp += sent + ". "
                    else:
                        if temp:
                            chunks.append(temp.strip())
                        temp = sent + ". "
                current_chunk = temp
            else:
                current_chunk = para

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

chunks = recursive_chunk(pdf_text, chunk_size=150, overlap=20)
print(f"Total chunks: {len(chunks)}")
for i, c in enumerate(chunks):
    print(f"\nChunk {i+1} ({len(c)} chars): {c}")

# Add to ChromaDB
for idx, chunk in enumerate(chunks):
    collection.add(
        documents=[chunk],
        ids=[f"chunk_{idx}"],
        embeddings=[model.encode(chunk).tolist()]
    )

# Test
query = "what is performance measure for checkers"
results = collection.query(
    query_embeddings=[model.encode(query).tolist()],
    n_results=1
)

print(f"\nQuery: {query}")
print(f"Retrieved: {results['documents'][0][0]}")
print("\nDay 07 Done - Recursive chunking preserves sentences")