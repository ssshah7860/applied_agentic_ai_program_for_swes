"""CLASS 03 — MODULE 5: RAG EVALUATION — INDEPENDENT LEARNING EXAMPLES

One standalone def per concept. No example calls another example.
Select EXAMPLE_TO_RUN near bottom (1–13, 'ALL', or None).
Most examples use pure Python and require no API key or database.
Examples labeled [API] require OPENAI_API_KEY and may incur API charges.

Optional install for API example:
    pip install langchain-openai langchain-core pydantic python-dotenv

All ground-truth labels and scoring examples are illustrative classroom data.
"""

# 1. RETRIEVAL vs GENERATION EVALUATION [NO API]
def example_01_retrieval_vs_generation():
    """Retrieval checks selected documents; generation checks the written answer."""
    retrieved_ids = ['T1', 'T3']
    relevant_ids = {'T1', 'T2'}
    answer = 'A password reset may resolve the sign-in issue.'
    print('Retrieved IDs:', retrieved_ids)
    print('Correct IDs (ground truth):', relevant_ids)
    print('Retrieval question: did we fetch the right tickets?')
    print('Generated answer:', answer)
    print('Generation question: is the answer grounded, complete, relevant?')


# 2. PRECISION@K [NO API]
def example_02_precision_at_k():
    """Fraction of top K retrieved items that are relevant."""
    retrieved = ['T1', 'T2', 'T3', 'T4', 'T5']
    relevant = {'T1', 'T2', 'T4', 'T5'}
    k = 5
    hits = sum(ticket in relevant for ticket in retrieved[:k])
    precision = hits / k
    print(f'Precision@{k} = {hits}/{k} = {precision:.2f}')  # 4/5 = .80


# 3. RECALL@K [NO API]
def example_03_recall_at_k():
    """Fraction of ALL known relevant items retrieved in the top K."""
    retrieved = ['T1', 'T4', 'T9']
    relevant = {'T1', 'T2', 'T3', 'T4'}
    hits = len(set(retrieved) & relevant)
    recall = hits / len(relevant)
    print(f'Recall@3 = {hits}/{len(relevant)} = {recall:.2f}')  # 2/4


# 4. F1 SCORE [NO API]
def example_04_f1_score():
    """F1 is the harmonic mean of precision and recall."""
    precision, recall = 0.80, 0.50
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(f'Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.3f}')


# 5. EVALUATION DATASET AND GROUND TRUTH [NO API]
def example_05_eval_dataset_ground_truth():
    """Each test question has known relevant IDs; compare predicted IDs to labels."""
    dataset = [
        {'question': 'Cannot log in', 'relevant': {'T1', 'T3'}, 'retrieved': ['T1', 'T2']},
        {'question': 'Database timeout', 'relevant': {'T5'}, 'retrieved': ['T5', 'T4']},
    ]
    for case in dataset:
        hits = len(set(case['retrieved']) & case['relevant'])
        print(case['question'], 'hits=', hits, 'precision=', hits / len(case['retrieved']))


# 6. MACRO-AVERAGE ACROSS QUERIES [NO API]
def example_06_macro_average():
    """Compute one score per query, then give each question equal weight."""
    import statistics
    cases = [
        {'retrieved': ['T1', 'T2'], 'relevant': {'T1'}},
        {'retrieved': ['T3', 'T4'], 'relevant': {'T3', 'T4'}},
    ]
    per_question = []
    for case in cases:
        precision = sum(t in case['relevant'] for t in case['retrieved']) / len(case['retrieved'])
        per_question.append(precision)
    print('Per-query precisions:', per_question)
    print('Macro Precision:', statistics.mean(per_question))  # (0.5 + 1.0) / 2


# 7. COMPARE DIFFERENT K VALUES [NO API]
def example_07_compare_k():
    """K is selected by us; examine precision/recall at K=1,3,5."""
    ranked = ['T1', 'T9', 'T2', 'T8', 'T3']
    relevant = {'T1', 'T2', 'T3'}
    for k in [1, 3, 5]:
        hits = sum(t in relevant for t in ranked[:k])
        precision = hits / k
        recall = hits / len(relevant)
        print(f'K={k}: P@K={precision:.3f}, R@K={recall:.3f}')


# 8. AVERAGE PRECISION — RANKING QUALITY [NO API]
def example_08_average_precision():
    """AP = sum of precision at relevant-hit ranks / total relevant count."""
    ranked = ['T1', 'T9', 'T2', 'T8', 'T3']
    relevant = {'T1', 'T2', 'T3'}
    hits = 0
    precision_sum = 0.0
    for rank, ticket in enumerate(ranked, start=1):
        if ticket in relevant:
            hits += 1
            precision_at_rank = hits / rank
            precision_sum += precision_at_rank
            print(f'Relevant at rank {rank}: Precision@{rank}={precision_at_rank:.3f}')
    ap = precision_sum / len(relevant) if relevant else 0.0
    print(f'Average Precision = {ap:.3f}')  # (1 + 2/3 + 3/5) / 3


# 9. LLM-AS-A-JUDGE + STRUCTURED OUTPUT [API]
def example_09_llm_as_judge():
    """Ask an LLM to rate groundedness, completeness and relevance from 1–5."""
    from pydantic import BaseModel, Field
    from langchain_openai import ChatOpenAI
    class JudgeScores(BaseModel):
        groundedness: int = Field(ge=1, le=5)
        completeness: int = Field(ge=1, le=5)
        relevance: int = Field(ge=1, le=5)
        explanation: str
    judge = ChatOpenAI(model='gpt-4o-mini', temperature=0).with_structured_output(JudgeScores)
    result = judge.invoke(
        'Evidence: Ticket T1: reset password to restore login.\n'
        'Question: How to restore login?\n'
        'Answer: Reset your password.\n'
        'Judge GROUNDEDNESS (supported by evidence), COMPLETENESS, RELEVANCE. '
        'Score 1 to 5; justify briefly.')
    print(result.model_dump())
    print('LLM judge ratings are subjective estimates, not verified truth.')


# 10. RETRIEVAL vs GENERATION FAILURE ANALYSIS [NO API]
def example_10_failure_analysis():
    """Classify whether missing evidence or unsupported wording caused failure."""
    cases = [
        {'name': 'A', 'relevant_retrieved': False, 'answer_supported': False},
        {'name': 'B', 'relevant_retrieved': True, 'answer_supported': False},
        {'name': 'C', 'relevant_retrieved': True, 'answer_supported': True},
    ]
    for case in cases:
        if not case['relevant_retrieved']:
            diagnosis = 'Retrieval failure: needed evidence missing.'
        elif not case['answer_supported']:
            diagnosis = 'Generation failure: evidence was available but answer unsupported.'
        else:
            diagnosis = 'No failure in this simplified check.'
        print(case['name'], diagnosis)


# 11. LATENCY + ESTIMATED API COST [NO API]
def example_11_latency_cost():
    """Measure real local elapsed time; token and price numbers are explicitly made up."""
    import time
    start = time.perf_counter()
    records = ['T1', 'T2', 'T3']  # local work, no network call
    _ = [ticket.lower() for ticket in records]
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f'Local operation latency: {elapsed_ms:.4f} ms')
    input_tokens, output_tokens = 1000, 200  # illustrative only
    input_price_per_million, output_price_per_million = 0.50, 2.00  # invented for math
    cost = (input_tokens * input_price_per_million + output_tokens * output_price_per_million) / 1_000_000
    print(f'Illustrative API cost calculation: ${cost:.6f} (NOT current provider pricing)')


# 12. EVALUATION REPORT [NO API]
def example_12_evaluation_report():
    """Build a small readable report from sample metrics."""
    report = [
        {'question': 'Login failure', 'precision': 0.8, 'recall': 0.5, 'grounded': True},
        {'question': 'Database timeout', 'precision': 1.0, 'recall': 1.0, 'grounded': False},
    ]
    for row in report:
        print(f"{row['question']}: P={row['precision']:.2f}, R={row['recall']:.2f}, grounded={row['grounded']}")
    macro_precision = sum(x['precision'] for x in report) / len(report)
    print('Macro precision:', macro_precision)


# 13. HALLUCINATION / CITATION CHECK [NO API]
def example_13_hallucination_detection():
    """Demonstrate explicit fact and citation checks (not a semantic proof)."""
    retrieved_ids = {'T1', 'T2'}
    answer_citations = ['T1', 'T99']
    unknown_citations = [x for x in answer_citations if x not in retrieved_ids]
    if unknown_citations:
        print('WARNING: Citation IDs absent from retrieved evidence:', unknown_citations)
    else:
        print('All citation IDs appear in retrieved evidence.')
    print('Citation presence does NOT prove every factual claim is grounded.')


EXAMPLE_TO_RUN = 1  # 1–13, 'ALL', None


def main():
    """Dispatcher only; every individual concept function is standalone."""
    examples = {
        1: example_01_retrieval_vs_generation,
        2: example_02_precision_at_k,
        3: example_03_recall_at_k,
        4: example_04_f1_score,
        5: example_05_eval_dataset_ground_truth,
        6: example_06_macro_average,
        7: example_07_compare_k,
        8: example_08_average_precision,
        9: example_09_llm_as_judge,
        10: example_10_failure_analysis,
        11: example_11_latency_cost,
        12: example_12_evaluation_report,
        13: example_13_hallucination_detection,
    }
    if EXAMPLE_TO_RUN is None:
        print('Set EXAMPLE_TO_RUN to 1–13 or ALL.')
        return
    selected = examples.items() if str(EXAMPLE_TO_RUN).upper() == 'ALL' else [(EXAMPLE_TO_RUN, examples[EXAMPLE_TO_RUN])]
    for number, example in selected:
        print(f'\n{"=" * 60}\nEXAMPLE {number}: {example.__name__}\n{"=" * 60}')
        example()


if __name__ == '__main__':
    main()
