import random

from tlm.mysql_sentence import get_sent
from nltk import word_tokenize

idf_cut = 1000
idf = NotImplemented

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


r = get_random_sent()
print(r)