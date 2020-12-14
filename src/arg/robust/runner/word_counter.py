import sys
from typing import Set

import nltk

from data_generator.tokenizer_wo_tf import get_tokenizer
from tab_print import print_table

gram_target = [4,3,2,1]


def get_ngrams(tokens):
    n_grams = []
    for i in gram_target:
        items = nltk.ngrams(tokens, i)
        n_grams.append(nltk.Counter(items))
    return n_grams


def main():
    text1 = open(sys.argv[1], "r").read()
    text2 = open(sys.argv[2], "r").read()

    tokenizer = get_tokenizer()

    tokens1 = tokenizer.tokenize(text1)
    tokens2 = tokenizer.tokenize(text2)

    rep1 = get_ngrams(tokens1)
    rep2 = get_ngrams(tokens2)

    for idx, (cnt1, cnt2) in enumerate(zip(rep1, rep2)):
        common: Set[str] = set(cnt1.keys()).intersection(cnt2.keys())
        rows = []
        for ngram in common:
            row = [ngram, cnt1[ngram], cnt2[ngram]]
            rows.append(row)

        print_table(rows)


if __name__ == "__main__":
    main()