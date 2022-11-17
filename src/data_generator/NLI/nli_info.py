import os

from cpath import data_path

num_classes = 3
corpus_dir = os.path.join(data_path, "nli")
tags = ["conflict", "match", "mismatch"]
labels = ["entailment", "neutral", "contradiction", ]


def nli_tokenized_path(split):
    return os.path.join(corpus_dir, "{}_tokenized.pickle".format(split))
