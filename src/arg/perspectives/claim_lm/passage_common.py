from arg.perspectives.clueweb_db import load_doc
from arg.perspectives.select_paragraph_claim import remove_duplicate


def score_passages(q_res, top_n, get_passage_score):
    passages = []
    docs = []
    for i in range(top_n):
        try:
            doc = load_doc(q_res[i].doc_id)
            docs.append(doc)
        except KeyError:
            pass
    for doc in remove_duplicate(docs):
        idx = 0
        window_size = 300
        while idx < len(doc):
            p = doc[idx:idx + window_size]
            score = get_passage_score(p)
            passages.append((p, score))
            idx += window_size
    return passages