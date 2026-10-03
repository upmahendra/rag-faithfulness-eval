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

# Day 04 - RAG Faithfulness Checker

## Objective
Implement a Faithfulness evaluation for RAG pipeline to detect hallucination. Check if LLM answer is faithful to retrieved context.

## What we did in Day 03 vs Day 04

**Day 03: Basic RAG Retrieval**
- Setup ChromaDB + SentenceTransformer (all-MiniLM-L6-v2)
- Add documents and retrieve top-1 context for query

**Day 04: Faithfulness Evaluation**
- Added `check_faithfulness()` function
- Used cosine similarity between answer and context
- Classify as FAITHFUL / NOT FAITHFUL based on threshold


