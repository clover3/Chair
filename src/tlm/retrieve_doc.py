from adhoc.bm25 import BM25_2
from collections import Counter

# def BM25_2(f, df, N, dl, avdl):




def retrive_docs(query):
    q_tf = Counter(query)
    doc_score = Counter()
    for term, qf in q_tf.items():
        postings = get_posting(term)
        qdf = len(postings)
        for doc_id, loc_list in postings:
            tf = len(loc_list)

            dl = get_doc_len(doc_id)
            total_doc = NotImplemented

            doc_score[doc_id] += BM25_2(tf, qdf, total_doc, dl, avdl)

