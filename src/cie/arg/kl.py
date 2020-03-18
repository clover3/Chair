from collections import Counter

import math

from list_lib import flatten
from models.classic.stopword import load_stopwords
from summarization import tokenizer


def kl_divergence_subset(a, b):
    tf_sum = sum(a.values())
    btf_sum = sum(b.values())
    kl = 0
    for term, tf in a.items():
        p_a = tf / tf_sum
        p_b = (b[term] + tf) / (btf_sum + tf)
        kl += -p_a * math.log(p_b / p_a)
    return kl


def kl_divergence(a, b):
    tf_sum = sum(a.values())
    btf_sum = sum(b.values())
    kl = 0

    all_terms = set(a.keys())
    all_terms.update(b.keys())
    l = len(all_terms)
    for term in all_terms:
        p_a = (a[term]+1) / (tf_sum + l)
        p_b = (b[term]+1) / (btf_sum+ l)
        kl += -p_a * math.log(p_b / p_a)
    return kl

class KLPredictor:
    def __init__(self, sents):
        self.stopwords = load_stopwords()
        self.topic_tf = Counter(flatten([self.tokenize(s) for s in sents]))
        self.threshold = None

    def tokenize(self, x):
        return tokenizer.tokenize(x, self.stopwords)

    def divergence(self, sent):
        return kl_divergence_subset(Counter(self.tokenize(sent)), self.topic_tf)


    def tune(self, sents, true_portion):
        scores = list([self.divergence(s) for s in sents])
        scores.sort()

        idx = int(len(scores) * true_portion)
        self.threshold = scores[idx]

    def predict(self, sent):
        return self.divergence(sent) < self.threshold
