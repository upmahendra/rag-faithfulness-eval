# RAG Faithfulness Evaluator

> Does your RAG answer actually come from retrieved documents or is it hallucinating?

I'm building this in public over 30 days. The goal is to detect when LLMs ignore context and make things up.

**GitHub:** https://github.com/upmahendra/rag-faithfulness-eval

---

## Why This?

Most RAG demos show if an answer *looks* correct. In production, we need to know if the answer is *faithful* to the documents it retrieved.

**Fluent answer!= Faithful answer.**

If the retrieved context says "Performance is % of games won" and the LLM says "Performance is accuracy and speed", it's hallucinating — even if it sounds fluent.

## What I Built So Far

### Day 01 - Setup
- Defined problem: Hallucination detection in RAG
- Planned 30-day roadmap: Retrieval -> Faithfulness -> Relevancy -> Metrics

### Day 02 - Making Text Searchable by Meaning
- Used `all-MiniLM-L6-v2` to convert text to 384-dim vectors
- Implemented semantic search with cosine similarity
- Query: `what is performance measure for checkers?` correctly retrieves checkers doc, not python doc

### Day 03 - RAG Retrieval with ChromaDB
- Setup ChromaDB vector store + SentenceTransformer
- Add 3 docs and retrieve top-1 context
- Output: `The checkers program learns... Performance is percent of games it wins`

### Day 04 - RAG Faithfulness Checker [Current]
Implemented `check_faithfulness()` to detect hallucination.

**Method:**
```python
def check_faithfulness(answer, context):
    a = model.encode(answer, normalize_embeddings=True)
    c = model.encode(context, normalize_embeddings=True)
    score = np.dot(a, c) # cosine similarity
    return "FAITHFUL" if score > 0.50 else "NOT FAITHFUL"
