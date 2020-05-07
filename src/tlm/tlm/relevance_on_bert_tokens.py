from collections import Counter

import math

from adhoc.bm25 import BM25_3
from data_generator.tokenizer_wo_tf import is_continuation, get_tokenizer
from models.classic.stopword import load_stopwords


def get_name(sb_indice):
    return " ".join([str(t) for t in sb_indice])

class Ranker:
    def __init__(self):
        self.tokenizer = get_tokenizer()

        self.stopwords = load_stopwords()
        self.df = Counter()
        self.N = 0

    def init_df(self, docs):
        # doc : list[int]
        dl_list = []
        self.df = Counter()
        for doc in docs:
            tf = self.get_terms(doc)
            dl_list.append(sum(tf.values()))
            for term in tf:
                self.df[term] += 1

        self.N = len(docs)
        self.avdl = sum(dl_list) / len(dl_list)

    def init_df_from_tf_list(self, tf_list):
        # doc : list[int]
        dl_list = []
        self.df = Counter()
        for tf in tf_list:
            dl_list.append(sum(tf.values()))
            for term in tf:
                self.df[term] += 1

        self.N = len(tf_list)
        self.avdl = sum(dl_list) / len(dl_list)


    def idf(self, key):
        n = self.df[key] + 1
        return math.log(self.N/n)

    def bm25(self, source_tokens_tf, target_tokens_tf, log=False):
        acc = 0
        dl = sum(target_tokens_tf.values())

        for key in source_tokens_tf:
            t = BM25_3(f= target_tokens_tf[key],
                   qf=source_tokens_tf[key],
                   df=self.df[key],
                   N=self.N,
                   dl=dl,
                   avdl=self.avdl
                   )
            acc += t
            if log:
                print(key, t)
        return acc

    def get_terms(self, tokens):
        word_list = []
        current_word = []
        #  : concatenate splitted subwords
        for i, t in enumerate(tokens):
            if is_continuation(t):
                current_word.append(t)
            else:
                if current_word:
                    word_list.append(current_word)
                current_word = [t]

        if current_word:
            word_list.append(current_word)

        word_list = ["".join(l) for l in word_list]

        def is_stop_word(w):
            if w in self.stopwords:
                return True
            elif self.N > 0 and self.df[w] > self.N * 0.5:
                return True
            else:
                return False
        #  drop stopwords
        words = [w for w in word_list if not is_stop_word(w)]
        tf = Counter(words)
        return tf

    def get_terms_from_ids(self, ids):
        return self.get_terms(self.tokenizer.convert_ids_to_tokens(ids))