from typing import List

from arg.perspectives.load import load_train_claim_ids, get_claims_from_ids
from misc_lib import split_7_3


def train_split():
    d_ids: List[int] = list(load_train_claim_ids())
    claims = get_claims_from_ids(d_ids)
    train, val = split_7_3(claims)
    return claims, val