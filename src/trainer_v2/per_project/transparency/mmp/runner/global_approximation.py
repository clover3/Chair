import random
from collections import Counter

from adhoc.kn_tokenizer import KrovetzNLTKTokenizer
from trainer_v2.per_project.transparency.mmp.data_enum import enum_pos_neg_sample


def f_prime(tf):
    # d s/d tf
    # gradient of score with respect to tf
    return NotImplemented


def main():
    random.seed(0)
    tokenizer = KrovetzNLTKTokenizer()

    def count_tf(text: str) -> Counter:
        tokens = tokenizer.tokenize_stem(text)
        return Counter(tokens)

    itr = enum_pos_neg_sample()
    pair_appear_cnt = Counter()
    accumulated_doc_len = 0
    pair_cnt = 0

    q_term_all = set()
    d_term_all = set()
    for q, d1, d2 in itr:
        tf_q = count_tf(q)
        tf_d1 = count_tf(d1)
        tf_d2 = count_tf(d2)
        s1 = 1
        s2 = 0
        loss = s1 - s2

        accumulated_doc_len += len(tf_d1)
        accumulated_doc_len += len(tf_d2)

        for q_term, q_cnt in tf_q.items():
            for term, cnt in tf_d1.items():
                grad = f_prime(cnt)
                pair_appear_cnt[q_term, term] += 1
                d_term_all.add(term)
            for term, cnt in tf_d2.items():
                grad = f_prime(cnt)
                d_term_all.add(term)
                pair_appear_cnt[q_term, term] += 1

            q_term_all.add(q_term)

        pair_cnt += 1

    before_compression = sum(pair_appear_cnt.values())
    after_compression = len(pair_appear_cnt)
    print("Number of (q,d1,d2) triplet", pair_cnt)
    print("Pair Appear Cnt", len(pair_appear_cnt))
    print("All QD pairs", before_compression)
    print("Compression rate", after_compression / before_compression)
    print("accumulated_doc_len: ", accumulated_doc_len)
    print("Unique Query terms", len(q_term_all))
    print("Unique Doc terms", len(d_term_all))
    print(" -> multiplied", len(q_term_all) * len(d_term_all))


if __name__ == "__main__":
    main()
