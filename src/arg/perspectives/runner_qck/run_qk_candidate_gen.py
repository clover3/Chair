from typing import List, Dict, Tuple

from arg.perspectives.load import get_claims_from_ids, load_claim_ids_for_split
from arg.qck.decl import QCKQuery, KDP, QKUnit
from arg.qck.kd_candidate_gen import qk_candidate_gen
from base_type import FilePath
from cache import save_to_pickle
from list_lib import lmap


def config1():
    return {
        'step_size': 300,
        'window_size': 300
    }


def config2():
    return {
        'step_size': 30,
        'window_size': 300
    }



def train_gen():
    split = "train"
    q_res_path = FilePath("/mnt/nfs/work3/youngwookim/data/perspective/train_claim/q_res_100")
    candidate = get_candidates(q_res_path, split, config1())
    save_to_pickle(candidate, "perspective_qk_candidate_train")


def train_gen_overlap():
    split = "train"
    q_res_path = FilePath("/mnt/nfs/work3/youngwookim/data/perspective/train_claim/q_res_100")
    candidate: List[QKUnit] = get_candidates(q_res_path, split, config2())
    save_to_pickle(candidate, "perspective_qk_candidate_train_dense")


def dev_gen():
    split = "dev"
    q_res_path = FilePath("/mnt/nfs/work3/youngwookim/data/perspective/dev_claim/q_res_100")
    candidate = get_candidates(q_res_path, split, config1())
    save_to_pickle(candidate, "perspective_qk_candidate_dev")


def get_candidates(q_res_path, split, config) -> List[QKUnit]:
    d_ids = list(load_claim_ids_for_split(split))
    claims: List[Dict] = get_claims_from_ids(d_ids)

    def claim_to_query(claim: Dict):
        return QCKQuery(str(claim['cId']), claim['text'])

    queries: List[QCKQuery] = lmap(claim_to_query, claims)
    top_n = 10
    candidate: List[Tuple[QCKQuery, List[KDP]]] = qk_candidate_gen(q_res_path, queries, top_n, config)
    return candidate


if __name__ == "__main__":
    train_gen_overlap()