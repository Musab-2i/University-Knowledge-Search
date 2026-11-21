from boolean_search import boolean_search
from ranked_search import ranked_search, explain_top_result
from output import print_boolean_results, print_ranked_results


def test_boolean(index, all_ids, documents_by_id):
    tests = [
        ("university", "AND", "student"),
        ("war", "OR", "king"),
        ("city", "NOT", "war"),        # city AND NOT war
        ("zzzzzz", "AND", "city"),     # كلمة غير موجودة -> نتيجة فاضية
    ]
    for t1, op, t2 in tests:
        print("\nQuery:", t1, op, t2)
        result = boolean_search(t1, op, t2, index, all_ids)
        print_boolean_results(result, documents_by_id)


def test_ranked(vectorizer, X, documents):
    queries = [
        "university students education",
        "what causes climate change",
        "history of the war",
    ]
    for q in queries:
        print("\nQuery:", q)
        results = ranked_search(q, vectorizer, X, documents, top_k=5)
        print_ranked_results(results)
        explain_top_result(q, results)


def run_all_tests(index, all_ids, documents_by_id, vectorizer, X, documents):
    test_boolean(index, all_ids, documents_by_id)
    test_ranked(vectorizer, X, documents)
