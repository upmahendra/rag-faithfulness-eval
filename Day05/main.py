# Day 05 Full RAGAS 3 Metrics Implementation
# pip install sentence-transformers scikit-learn

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading embedding model")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded")

def check_relevancy(a, b):
    if not a.strip() or not b.strip():
        return 0.0
    embeddings = model.encode([a, b])
    score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return round(float(score), 4)

def check_faithfulness(answer, context):
    return check_relevancy(answer, context)

def evaluate_rag(query, context, answer):
    print(f"Query {query}")
    print(f"Context {context}")
    print(f"Answer {answer}")
    print("")

    faith_score = check_faithfulness(answer, context)
    context_score = check_relevancy(query, context)
    answer_score = check_relevancy(query, answer)

    print(f"Faithfulness answer to context {faith_score}")
    print(f"Context Relevancy query to context {context_score}")
    print(f"Answer Relevancy query to answer {answer_score}")
    print("")

    if faith_score > 0.5 and context_score > 0.5 and answer_score > 0.5:
        print("Verdict GOOD RAG")
    else:
        print("Verdict BAD RAG needs fix")
        if context_score <= 0.5:
            print("Fix Retrieval failed context not relevant")
        if answer_score <= 0.5:
            print("Fix Answer is off topic")
        if faith_score <= 0.5:
            print("Fix Hallucination answer not from context")

    return faith_score, context_score, answer_score

if __name__ == "__main__":
    query = "what is performance measure for checkers"
    context = "The checkers program performance measure is percent of games won It plays against opponents and winning rate is calculated"
    answer = "Performance is percent of games won in checkers"

    evaluate_rag(query, context, answer)

    print("")
    print("Testing BAD RAG")
    print("")

    bad_context = "The capital of France is Paris and Eiffel Tower is there"
    bad_answer = "Paris is the capital of France"
    evaluate_rag(query, bad_context, bad_answer)