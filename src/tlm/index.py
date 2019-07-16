import data_generator.data_parser.trec as trec
import nltk.tokenize
from krovetzstemmer import Stemmer
import path
import os
import pickle
from models.classic.stopword import load_stopwords
from misc_lib import TimeEstimator


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


if __name__ == "__main__":
    save_doc_len()
    #build_krovetz_index()
