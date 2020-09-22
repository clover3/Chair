from typing import List

from arg.qck.decl import QKUnit
from cache import load_from_pickle


# Unfiltered candidates from top BM25
def load_qk_candidate_train() -> List[QKUnit]:
    return load_from_pickle("perspective_qk_candidate_train")


# Unfiltered candidates from top BM25
def load_qk_candidate_dev() -> List[QKUnit]:
    return load_from_pickle("perspective_qk_candidate_dev")