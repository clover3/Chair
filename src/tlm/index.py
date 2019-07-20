import data_generator.data_parser.trec as trec
import nltk.tokenize
from krovetzstemmer import Stemmer
import path
import os
import pickle
from models.classic.stopword import load_stopwords
from misc_lib import TimeEstimator
from collections import Counter

class CacheStemmer:
    def __init__(self):
        self.stemmer = Stemmer()
        self.stem_dict = dict()

    def stem(self, token):
        if token in self.stem_dict:
            return self.stem_dict[token]
        else:
            r = self.stemmer.stem(token)
            self.stem_dict[token] = r
            return r

def stemmed_counter(tokens, stemmer):
    c = Counter()
    for t in tokens:
        c[stemmer.stem(t)] += 1

    return c


def build_krovetz_index():
    stemmer = Stemmer()
    stopwords = load_stopwords()

    stem_dict = dict()

    def stem(token):
        if token in stem_dict:
            return stem_dict[token]
        else:
            r = stemmer.stem(token)
            stem_dict[token] = r
            return r

    collection = trec.load_robust(trec.robust_path)
    print("writing...")
    inv_index = dict()
    ticker = TimeEstimator(len(collection))

    for doc_id in collection:
        content = collection[doc_id]
        tokens = nltk.tokenize.wordpunct_tokenize(content)
        terms = dict()
        for idx, t in enumerate(tokens):
            if t in stopwords:
                continue

            t_s = stem(t)

            if t_s not in terms:
                terms[t_s] = list()

            terms[t_s].append(idx)


        for t_s in terms:
            if t_s not in inv_index:
                inv_index[t_s] = list()

            posting = (doc_id, terms[t_s])
            inv_index[t_s].append(posting)

        ticker.tick()

    save_path = os.path.join(path.data_path, "adhoc", "robust_inv_index.pickle")
    pickle.dump(inv_index, open(save_path, "wb"))


def save_doc_len():
    collection = trec.load_robust(trec.robust_path)
    print("writing...")
    ticker = TimeEstimator(len(collection))

    doc_len = dict()
    for doc_id in collection:
        content = collection[doc_id]
        tokens = nltk.tokenize.wordpunct_tokenize(content)
        doc_len[doc_id] = len(tokens)
        ticker.tick()

    save_path = os.path.join(path.data_path, "adhoc", "doc_len.pickle")
    pickle.dump(doc_len, open(save_path, "wb"))



def save_qdf():
    ii_path = os.path.join(path.data_path, "adhoc", "robust_inv_index.pickle")
    inv_index = pickle.load(open(ii_path, "rb"))
    qdf_d = Counter()
    for term in inv_index:
        qdf = len(inv_index[term])
        qdf_d[term] = qdf

    save_path = os.path.join(path.data_path, "adhoc", "robust_qdf.pickle")
    pickle.dump(qdf_d, open(save_path, "wb"))

def save_qdf_ex():
    ii_path = os.path.join(path.data_path, "adhoc", "robust_inv_index.pickle")
    inv_index = pickle.load(open(ii_path, "rb"))
    save_path = os.path.join(path.data_path, "adhoc", "robust_meta.pickle")
    meta = pickle.load(open(save_path, "rb"))
    stopwords = load_stopwords()
    stemmer = CacheStemmer()

    simple_posting = {}

    qdf_d = Counter()
    for term in inv_index:
        simple_posting[term] = set()
        for doc_id, _ in inv_index[term]:
            simple_posting[term].add(doc_id)

    for doc in meta:
        date, headline = meta[doc]
        tokens = nltk.tokenize.wordpunct_tokenize(headline)
        terms = set()
        for idx, t in enumerate(tokens):
            if t in stopwords:
                continue

            t_s = stemmer.stem(t)

            terms.add(t_s)

        for t in terms:
            simple_posting[t].add(doc)

    for term in inv_index:
        qdf = len(simple_posting[term])
        qdf_d[term] = qdf

    save_path = os.path.join(path.data_path, "adhoc", "robust_qdf_ex.pickle")
    pickle.dump(qdf_d, open(save_path, "wb"))


def save_title():
    collection = trec.load_robust_meta(trec.robust_path)
    save_path = os.path.join(path.data_path, "adhoc", "robust_meta.pickle")
    pickle.dump(collection, open(save_path, "wb"))



if __name__ == "__main__":
    save_qdf_ex()
    #build_krovetz_index()
