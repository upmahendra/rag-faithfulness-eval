# rag-faithfulness-eval

day 01 — i found llm doesn't stick to source. asked about checkers program, book says "world tournament", chatgpt says "opponents" + adds extra theory.

that's hallucination. building toolkit to measure it.

stack: python, chroma, ragas

day 01 done — check Day01/lab_book.md

# Day 02 — Making Text Searchable

Today I finally understood what RAG's "R" actually means.

I installed ChromaDB and sentence-transformers and tried to make my laptop understand text like humans do.

I gave it 3 lines:
- checkers program
- machine learning definition
- python definition

It converted each line into 384 numbers. That's called an embedding. It’s how AI remembers meaning, not just words.

Then I asked it: "what is performance measure for checkers?"

It didn't do keyword matching. It compared the meaning of my question with those 384-number vectors and returned the exact checkers line.

That moment clicked for me — retrieval is not search, it's meaning matching.

Next: Check if the answer stays faithful to what we retrieved.
