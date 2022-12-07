from collections import Counter
from typing import List

from arg.perspectives.load import get_claims_from_ids, load_dev_claim_ids, claims_to_dict
from arg.perspectives.kn_tokenizer import KrovetzNLTKTokenizer
from arg.perspectives.random_walk.runner.show_biased_random_walk import normalize_counter_to_sum1
from cache import load_from_pickle, save_to_pickle


def pc_new_init_prob():
    d_ids: List[int] = list(load_dev_claim_ids())
    claims = get_claims_from_ids(d_ids)
    claim_d = claims_to_dict(claims)
    bias_plus_word: Counter = load_from_pickle("bias_plus_words")
    tokenizer = KrovetzNLTKTokenizer()

    base_p = max(bias_plus_word.values())

    init_p_score_d = {}
    for cid in d_ids:
        c_text = claim_d[cid]
        tokens = tokenizer.tokenize_stem(c_text)

        score_for_cid = Counter()
        for t, cnt in Counter(tokens).items():
            prob = cnt * base_p
            score_for_cid[t] = prob

        for t, score in bias_plus_word.items():
            score_for_cid[t] += score

        score_for_cid = normalize_counter_to_sum1(score_for_cid)
        init_p_score_d[cid] = score_for_cid

    save_to_pickle(init_p_score_d, "pc_dev_new_init_prob")


pc_new_init_prob()





