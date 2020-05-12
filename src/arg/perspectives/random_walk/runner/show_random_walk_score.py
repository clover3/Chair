from typing import List

from arg.perspectives.load import get_claims_from_ids, load_dev_claim_ids, claims_to_dict
from cache import load_from_pickle


def show_random_walk_score():
    d_ids: List[int] = list(load_dev_claim_ids())
    claims = get_claims_from_ids(d_ids)
    claim_d = claims_to_dict(claims)

    top_k = 7
    q_tf_replace = dict(load_from_pickle("bias_random_walk_dev_plus_all"))

    for claim_id, qtf in q_tf_replace.items():
        print(claim_d[claim_id])
        print(qtf.most_common(100))
    print("")


if __name__ == "__main__":
    show_random_walk_score()
