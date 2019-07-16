from adhoc.bm25 import BM25_2
from collections import Counter
from misc_lib import tprint
import pickle, os, path
from rpc.text_reader import TextReaderClient

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
        return self.inv_index[term]


    # query should be stemmed
    def retrieve_docs(self, query):
        q_tf = Counter(query)
        doc_score = Counter()
        for term, qf in q_tf.items():
            postings = self.get_posting(term)
            qdf = len(postings)
            for doc_id, loc_list in postings:
                tf = len(loc_list)

                dl = self.doc_len_dict[doc_id]
                total_doc = self.total_doc_n

                doc_score[doc_id] += BM25_2(tf, qdf, total_doc, dl, self.avdl)

        return doc_score.most_common(100)



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