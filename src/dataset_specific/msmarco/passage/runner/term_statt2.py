import itertools
from collections import Counter

import nltk as nltk

from adhoc.kn_tokenizer import count_df
from cache import save_to_pickle
from dataset_specific.msmarco.passage.passage_resource_loader import load_msmarco_collection
from misc_lib import get_second, TELI, TimeEstimator


def main():
    itr = load_msmarco_collection()
    size = 8841823
    tikcer = TimeEstimator(size)
    print("Build un-stemmed word count")
    df = Counter()
    tf = 0
    for _, text in itr:
        tikcer.tick()
        tokens = nltk.tokenize.word_tokenize(text)
        tf += len(tokens)
        for term in set(tokens):
            df[term] += 1

    save_to_pickle(df, "msmarco_passage_unstemmed_df")


if __name__ == "__main__":
    main()