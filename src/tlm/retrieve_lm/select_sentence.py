import random

from tlm.retrieve_lm.mysql_sentence import get_sent
from nltk import word_tokenize
from cache import load_from_pickle

idf_cut = 8
idf = load_from_pickle("robust_idf_mini")

def get_random_sent():
    def good(r):
        sent = r[4]
        if len(sent)< 10:
            return False

        tokens = word_tokenize(sent)
        max_idf = max([idf[t] for t in tokens])
        if max_idf < idf_cut:
            return False

        return True
    n_sentence = 1000 * 10000
    r = None
    while r is None or not good(r):
        idx = random.randint(1, n_sentence)
        r = get_sent(idx)
    return r
