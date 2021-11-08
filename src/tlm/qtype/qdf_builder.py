from collections import Counter

import nltk

from cache import save_to_pickle
from dataset_specific.msmarco.common import load_queries


def build_save_qdf():
    queries = load_queries("train")

    df = Counter()
    for qid, q_str in queries:
        q_tokens = nltk.word_tokenize(q_str)
        for token in set(q_tokens):
            df[token] += 1

    for term, cnt in df.most_common(200):
        print(term, cnt)

    save_to_pickle(df, "msmarco_qdf")



def main():
    queries = load_queries("train")
    n_list = [1, 2, 3, 4, 5]
    all_n_grams = {n: Counter() for n in n_list}

    df = Counter()
    for qid, q_str in queries:
        q_tokens = nltk.word_tokenize(q_str)
        for n in n_list:
            ngram_list = nltk.ngrams(q_tokens, n)
            all_n_grams[n].update(ngram_list)

    save_to_pickle(all_n_grams, "msmarco_ngram_qdf")
    # for token in set(q_tokens):
    #     df[token] += 1

    for n in n_list:
        for n_gram, cnt, in all_n_grams[n].most_common(200):
            print(n_gram, cnt)


if __name__ == "__main__":
    main()