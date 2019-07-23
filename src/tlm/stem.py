from collections import Counter
from krovetzstemmer import Stemmer


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