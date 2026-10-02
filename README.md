# RAG Faithfulness Evaluator

> Does your RAG answer actually come from retrieved documents or is it hallucinating?

I am building this in public over 30 days. The goal is to detect when LLMs ignore context and make things up.

GitHub: https://github.com/upmahendra/rag-faithfulness-eval

## Why This?

Most RAG demos show if answer *looks* correct. In production, we need to know if answer is *faithful* to the documents it retrieved.

Fluent answer!= Faithful answer.

## What I Built So Far

### Day 01 - Setup
- Defined problem: Hallucination detection in RAG
- Planned 30-day roadmap

### Day 02 - Making Text Searchable by Meaning
- Used `all-MiniLM-L6-v2` to convert text to 384-dim vectors
- Implemented semantic search with cosine similarity
- Query: "what is performance measure for checkers?" correctly retrieves checkers doc, not python doc

### Day 03 - Simple Faithfulness Scoring
- Built first hallucination detector
- Method: Cosine similarity between answer and retrieved context
- Result:
    - Faithful answer: 0.59 -> FAITHFUL
    - Hallucinated answer: 0.35 -> NOT FAITHFUL
- No OpenAI key needed, pure embeddings
