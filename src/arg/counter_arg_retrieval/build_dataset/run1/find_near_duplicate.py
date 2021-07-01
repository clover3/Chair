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


def unigram_overlap(d1: str, d2: str):
    tokens1 = d1.split()
    tokens2 = d2.split()
    min_tf_sum, overlap_tf = get_tokens_overlap(tokens1, tokens2)
    print(overlap_tf/min_tf_sum)
    return overlap_tf, min_tf_sum


def get_ngram_count_rep(text):
    tokens = text.split()
    items = list(nltk.ngrams(tokens, 3))
    return Counter(items)



def ngram_overlap(d1: str, d2: str):
    def get_ngram(text):
        tokens = text.split()
        items = list(nltk.ngrams(tokens, 3))
        print(len(items))
        return items

    tf_common, tf_min = get_tokens_overlap(get_ngram(d1), get_ngram(d2))
    print(tf_common, tf_min)


def get_tokens_overlap(tokens1, tokens2):
    tf1 = Counter(tokens1)
    tf2 = Counter(tokens2)
    tf_sum1 = sum(tf1.values())
    tf_sum2 = sum(tf2.values())
    min_tf_sum = min(tf_sum1, tf_sum2)
    overlap_tf = 0
    for key in tf1:
        overlap_tf += min(tf1[key], tf2[key])
    return overlap_tf, min_tf_sum


def debug():
    doc_id1 = "clueweb12-0610wb-68-32030"
    doc_id2 = "clueweb12-0606wb-83-01024"
    docs = load_run1_doc_indexed()

    d1 = docs[doc_id1]
    d2 = docs[doc_id2]

    print(unigram_overlap(d1, d2))
    print(ngram_overlap(d1, d2))


def main():
    debug()


if __name__ == "__main__":
    main()


