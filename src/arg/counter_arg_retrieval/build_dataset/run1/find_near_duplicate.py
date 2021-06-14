from collections import Counter

import nltk

from arg.counter_arg_retrieval.build_dataset.resources import load_run1_doc_indexed
from arg.counter_arg_retrieval.build_dataset.run1.select_common_topk import top_k_combine_by_or


def text_to_ngram_counter(text, n=3):
    tokens = text.split()
    ngram_list = nltk.ngrams(tokens, n)
    return Counter(ngram_list)


def load_docs():
    target_doc_ids = top_k_combine_by_or()
    return get_duplicate_list(target_doc_ids)


def get_duplicate_list(target_doc_ids):
    docs = load_run1_doc_indexed()
    hash_set = []
    hash_to_d = {}
    skip_list = []
    for doc_id in target_doc_ids:
        text = docs[doc_id]
        h = hash(text)
        if h in hash_set:
            skip_list.append(doc_id)
        hash_set.append(h)
        hash_to_d[h] = doc_id
    return skip_list


def main():
    load_docs()
    return NotImplemented


if __name__ == "__main__":
    main()


