from collections import Counter
from typing import List

import math
from arg.perspectives.inspect import pc_predict_to_inspect

from arg.perspectives.bm25_predict import predict_by_bm25, get_bm25_module
from arg.perspectives.evaluate import inspect
from arg.perspectives.load import load_dev_claim_ids, get_claims_from_ids
from arg.perspectives.split_helper import train_split
from cache import load_from_pickle
from list_lib import dict_key_map


def run_bm25():
    claims, val = train_split()
    top_k = 20
    pred = predict_by_bm25(get_bm25_module(), claims, top_k)

    inspect(pred)


def get_idf():
    idf = load_from_pickle("robust_idf_mini")
    out_df = Counter()
    for key, value in idf.df.items():
        out_df[key.lower()] += value

    N = math.exp(idf.default_idf)
    return out_df, N * 2


def run_random_walk_score():
    d_ids: List[int] = list(load_dev_claim_ids())
    claims = get_claims_from_ids(d_ids)
    top_k = 20
    bm25 = get_bm25_module()
    #df, N = get_idf()
    #bm25.df = df
    #bm25.N = N
    q_tf_replace_0 = dict(load_from_pickle("random_walk_score_100"))
    q_tf_replace = dict(load_from_pickle("dev_claim_random_walk_debug2"))
    q_tf_replace = dict_key_map(lambda x: int(x), q_tf_replace)
    pc_predict_to_inspect(bm25, q_tf_replace, q_tf_replace_0,  claims, top_k)


if __name__ == "__main__":
    run_random_walk_score()
