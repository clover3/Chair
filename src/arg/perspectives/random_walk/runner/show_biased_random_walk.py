from collections import Counter
from typing import List

from arg.perspectives.load import get_claims_from_ids, claims_to_dict, load_train_claim_ids
from cache import load_from_pickle, save_to_pickle
from models.classic.stopword import load_stopwords


def normalize_counter_to_sum1(c: Counter) -> Counter:
    factor = 1 / sum(c.values())
    out_c = Counter()
    for key, value in c.items():
        out_c[key] = value * factor

    return out_c


def show_random_walk_score():
    d_ids: List[int] = list(load_train_claim_ids())
    claims = get_claims_from_ids(d_ids)
    claim_d = claims_to_dict(claims)

    prob_score_d = load_from_pickle("pc_pos_word_prob_train")
    stopwords = load_stopwords()
    acc_counter_prob_init = Counter()
    for claim_id, prob_scores in prob_score_d.items():
        for k, v in prob_scores:
            if k not in stopwords:
                acc_counter_prob_init[k] += v

    rw_score = dict(load_from_pickle("bias_random_walk_train_all"))
    acc_counter = Counter()
    for claim_id, qtf in rw_score.items():
        for k, v in qtf.items():
            acc_counter[k] += v

    acc_counter_prob_init = normalize_counter_to_sum1(acc_counter_prob_init)
    acc_counter = normalize_counter_to_sum1(acc_counter)

    new_counter = Counter()
    for k, v in acc_counter.items():
        if len(k) > 2:
            new_v = v - acc_counter_prob_init[k]
            new_counter[k] = new_v

    save_to_pickle(new_counter, "bias_plus_words")


if __name__ == "__main__":
    show_random_walk_score()
