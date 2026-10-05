# RAG Faithfulness Evaluator
Does your RAG answer actually come from retrieved documents or is it hallucinating?

I'm building this in public over 30 days. The goal is to detect when LLMs ignore context and make things up.

GitHub: https://github.com/upmahendra/rag-faithfulness-eval

Why This?
Most RAG demos show if an answer looks correct. In production, we need to know if the answer is faithful to the documents it retrieved.

Fluent answer!= Faithful answer.

If the retrieved context says "Performance is % of games won" and the LLM says "Performance is accuracy and speed", it's hallucinating — even if it sounds fluent.

What I Built So Far
Day 01 - Setup
Defined problem: Hallucination detection in RAG
Planned 30-day roadmap: Retrieval -> Faithfulness -> Relevancy -> Metrics

Day 02 - Making Text Searchable by Meaning
Used all-MiniLM-L6-v2 to convert text to 384-dim vectors
Implemented semantic search with cosine similarity
Query: what is performance measure for checkers? correctly retrieves checkers doc, not python doc

Day 03 - RAG Retrieval with ChromaDB
Setup ChromaDB vector store + SentenceTransformer
Add 3 docs and retrieve top-1 context
Output: The checkers program learns... Performance is percent of games it wins

Day 04 - RAG Faithfulness Checker
Implemented check_faithfulness() to detect hallucination.

Method:
def check_faithfulness(answer, context):
    a = model.encode(answer, normalize_embeddings=True)
    c = model.encode(context, normalize_embeddings=True)
    score = np.dot(a, c)
    return "FAITHFUL" if score > 0.50 else "NOT FAITHFUL"

Day 05 - Context Relevancy + Answer Relevancy [Current]
Implemented full RAGAS triad - 3 metrics evaluation.

Method:
def check_relevancy(a, b):
    emb = model.encode([a, b])
    score = cosine_similarity([emb[0]], [emb[1]])[0][0]
    return score

faith_score = check_faithfulness(answer, context)
context_score = check_relevancy(query, context)
answer_score = check_relevancy(query, answer)

if faith_score > 0.5 and context_score > 0.5 and answer_score > 0.5:
    print("GOOD RAG")
else:
    print("BAD RAG - needs fix")

Output:
Query: what is performance measure for checkers
Faithfulness: 0.8507 Context: 0.7034 Answer: 0.6688 = GOOD RAG
Bad Context Test: 0.7829, 0.0111, 0.0231 = BAD RAG needs fix
