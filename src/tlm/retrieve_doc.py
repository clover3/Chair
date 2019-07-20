from adhoc.bm25 import BM25_3, BM25_3_q_weight
from collections import Counter
from misc_lib import tprint
import pickle, os, path
from rpc.text_reader import TextReaderClient
from misc_lib import left

# def BM25_2(f, df, N, dl, avdl):


class RobustCollection:
    def __init__(self):
        ii_path = os.path.join(path.data_path, "adhoc", "robust_inv_index.pickle")
        self.inv_index = pickle.load(open(ii_path, "rb"))
        dl_path = os.path.join(path.data_path, "adhoc", "doc_len.pickle")
        self.doc_len_dict = pickle.load(open(dl_path, "rb"))
        self.total_doc_n =  len(self.doc_len_dict)
        self.avdl = sum(self.doc_len_dict.values()) / len(self.doc_len_dict)


    def get_posting(self, term):
        if term in self.inv_index:
            return self.inv_index[term]
        else:
            return []

    def high_idf_q_terms(self, q_tf, n_limit=10):
        total_doc = self.total_doc_n

        high_qt = Counter()
        for term, qf in q_tf.items():
            postings = self.get_posting(term)
            qdf = len(postings)
            w = BM25_3_q_weight(qf, qdf, total_doc)
            high_qt[term] = w

        return set(left(high_qt.most_common(n_limit)))



    # query should be stemmed
    def retrieve_docs(self, query, top_k = 100):
        q_tf = Counter(query)
        doc_score = Counter()
        high_qt = self.high_idf_q_terms(q_tf)
        for term, qf in q_tf.items():
            if term not in high_qt:
                continue

            postings = self.get_posting(term)
            qdf = len(postings)
            for doc_id, loc_list in postings:
                tf = len(loc_list)
                dl = self.doc_len_dict[doc_id]
                total_doc = self.total_doc_n
                doc_score[doc_id] += BM25_3(tf, qf, qdf, total_doc, dl, self.avdl)

        return doc_score.most_common(top_k)


if __name__ == "__main__":
    tprint("Initialize")
    col = RobustCollection()
    text_reader = TextReaderClient()
    query = "cis hard currency".split()

    tprint("Retrieving...")
    r = col.retrieve_docs(query)
    tprint("Done")

    for doc_id in r:
        print(doc_id)
        content = text_reader.retrieve(doc_id)
        print(content)
        break
