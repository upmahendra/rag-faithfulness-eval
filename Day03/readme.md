# Day 03 - Faithfulness Check Using Embeddings

## What I did today

Built a simple hallucination detector that tells if an LLM answer actually comes from the context.

## Idea

If answer is faithful, its meaning should be close to context meaning.
We can measure "closeness" using cosine similarity of embeddings.

## Example I tested

Context:
> The checkers program learns to play checkers. Performance is percent of games it wins in world tournament.

Test 1 - Faithful:
> Performance measure is percent of games it wins in world tournament.
> Score: 0.59 -> FAITHFUL

Test 2 - Hallucinated:
> Performance measure is accuracy and speed of python code.
> Score: 0.35 -> NOT FAITHFUL

Difference 0.24 is enough to flag hallucination.

## Code

```python
def check_faithfulness(answer, context):
    v1 = model.encode([answer])
    v2 = model.encode([context])
    return cosine_similarity(v1, v2)[0][0]