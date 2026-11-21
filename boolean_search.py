#Prepared by Mohammed Al-Hello
from preprocessing import preprocess


def docs_with(term, index):
    terms = preprocess(term)
    if len(terms) == 0:
        return set()
    return index.get(terms[0], set())


def boolean_search(term1, operator, term2, index, all_ids):
    A = docs_with(term1, index)
    if operator == "AND":
        return A & docs_with(term2, index)
    elif operator == "OR":
        return A | docs_with(term2, index)
    elif operator == "NOT":
        if term2 is None:
            return all_ids - A
        return A - docs_with(term2, index)
    return set()


def parse_boolean_query(query):
    words = query.split()
    upper = [w.upper() for w in words]

    if len(words) == 2 and upper[0] == "NOT":
        return (words[1], "NOT", None)

    if len(words) == 3 and upper[1] in ("AND", "OR"):
        return (words[0], upper[1], words[2])

    if len(words) == 4 and upper[1] == "AND" and upper[2] == "NOT":
        return (words[0], "NOT", words[3])

    return None
