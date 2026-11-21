def short_text(text, length=60):
    if len(text) <= length:
        return text
    return text[:length].rstrip() + " ..."


def print_boolean_results(doc_ids, documents_by_id, limit=10):
    ordered = sorted(doc_ids, key=lambda d: int(d[1:]))
    print("\nMatching Documents (" + str(len(ordered)) + "):")
    if len(ordered) == 0:
        print("- No matching documents.")
    for doc_id in ordered[:limit]:
        doc = documents_by_id[doc_id]
        print("- " + doc_id + " [" + doc["title"] + "]: " + short_text(doc["text"]))
    if len(ordered) > limit:
        print("... and", len(ordered) - limit, "more")


def print_ranked_results(results):
    if len(results) == 0:
        print("\nNo results found.")
        return
    print("\nTop-" + str(len(results)) + " Ranked Results (TF-IDF + Cosine):")
    for rank, (doc, score) in enumerate(results, start=1):
        print(str(rank) + ". " + doc["id"] + " (" + str(round(score, 2)) + ") ["
              + doc["title"] + "]: " + short_text(doc["text"]))
