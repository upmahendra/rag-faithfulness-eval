# RAG Faithfulness Evaluator
Does your RAG answer actually come from retrieved documents or is it hallucinating?

I'm building this in public over 30 days. The goal is to detect when LLMs ignore context and make things up.

GitHub: https://github.com/upmahendra/rag-faithfulness-eval

Why This?
Most RAG demos show if an answer looks correct. In production, we need to know if the answer is faithful to the documents it retrieved.

Fluent answer!= Faithful answer.

If the retrieved context says "Performance is % of games won" and the LLM says "Performance is accuracy and speed", it's hallucinating — even if it sounds fluent.

What I Built So Far
## Day 01 - Setup
Defined problem: Hallucination detection in RAG
Planned 30-day roadmap: Retrieval -> Faithfulness -> Relevancy -> Metrics

## Day 02 - Making Text Searchable by Meaning
Used all-MiniLM-L6-v2 to convert text to 384-dim vectors
Implemented semantic search with cosine similarity
Query: what is performance measure for checkers? correctly retrieves checkers doc, not python doc

## Day 03 - RAG Retrieval with ChromaDB
Setup ChromaDB vector store + SentenceTransformer
Add 3 docs and retrieve top-1 context
Output: The checkers program learns... Performance is percent of games it wins

## Day 04 - RAG Faithfulness Checker
Implemented check_faithfulness() to detect hallucination.
Method:
def check_faithfulness(answer, context):
    a = model.encode(answer, normalize_embeddings=True)
    c = model.encode(context, normalize_embeddings=True)
    score = np.dot(a, c)
    return "FAITHFUL" if score > 0.50 else "NOT FAITHFUL"

## Day 05 - Context Relevancy + Answer Relevancy
Implemented full RAGAS triad - 3 metrics evaluation.
Method:
def check_relevancy(a, b):
    emb = model.encode([a, b])
    score = cosine_similarity([emb[0]], [emb[1]])[0][0]
    return score
Output:
GOOD RAG: 0.8507, 0.7034, 0.6688 = GOOD RAG
BAD RAG: 0.7829, 0.0111, 0.0231 = BAD RAG needs fix

## Day 06 - Document Chunking [Current]
Problem: Real docs are long, can't embed whole doc as one vector.
Implemented fixed-size chunking with overlap.
Method:
def chunk_text(text, chunk_size=20, overlap=5):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks
Output:
No Overlap chunks: 3
Overlap chunks: 4
Overlap keeps context: "performance measure P is percent of games won" stays together
Query: what is performance measure for checkers
Top contexts correctly retrieves chunk with performance measure

## Day 07 - Recursive Chunking [Current]
Problem: Fixed word count breaks sentences in middle and loses meaning.
Ex: "Performance measure P is percent of" | "games won..." - Context broken.

Implemented recursive hierarchy: Paragraph (\n\n) -> Sentence (. ) -> Word.

Method:
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
            if len(para) > chunk_size:
                sentences = para.split(". ")
                temp = ""
                for sent in sentences:
                    if len(temp) + len(sent) <= chunk_size:
                        temp += sent + ". "
                    else:
                        chunks.append(temp.strip())
                        temp = sent + ". "
                current_chunk = temp
            else:
                current_chunk = para
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

Output:
Total chunks: 5
Chunk 1 (34 chars): Machine learning is a field of AI.
Chunk 3 (169 chars): The task T is playing checkers. The performance measure P is percent of games won...
Query: what is performance measure for checkers
Retrieved: The task T is playing checkers. The performance measure P is percent of games won in the world tournament...
Day 07 Done - Recursive chunking preserves sentences

Why Better:
Day 06: 12 chunks, breaks in middle
Day 07: 5 chunks, keeps full sentences + paragraphs together. This is how LangChain's RecursiveCharacterTextSplitter works internally.
