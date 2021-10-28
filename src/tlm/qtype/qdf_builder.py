from collections import Counter

import nltk

from cache import save_to_pickle
from dataset_specific.msmarco.common import load_queries


def main():
    queries = load_queries("train")

    df = Counter()
    for qid, q_str in queries:
        q_tokens = nltk.word_tokenize(q_str)
        for token in set(q_tokens):
            df[token] += 1

    for term, cnt in df.most_common(200):
        print(term, cnt)

    save_to_pickle(df, "msmarco_qdf")
    

if __name__ == "__main__":
    main()