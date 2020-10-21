from typing import List, Tuple

from arg.perspectives.load import splits
from arg.perspectives.qck.qck_common import get_qck_queries
from arg.qck.decl import QCKQuery, KDP, QKUnit
from arg.qck.kd_candidate_gen import qk_candidate_gen
from base_type import FilePath
from cache import save_to_pickle


def config1():
    return {
        'step_size': 300,
        'window_size': 300,
        'top_n': 10,

    }


def config2():
    return {
        'step_size': 30,
        'window_size': 300,
        'top_n': 10,
    }


def config3():
    return {
        'step_size': 300,
        'window_size': 300,
        'top_n': 100,

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
    queries = get_qck_queries(split)
    top_n = config['top_n']
    candidate: List[Tuple[QCKQuery, List[KDP]]] = qk_candidate_gen(q_res_path, queries, top_n, config)
    return candidate


def run_get_for_token_scoring():
    for split in splits:
        gen_for_token_scoring(split)


def gen_for_token_scoring(split):
    q_res_path = FilePath("/mnt/nfs/work3/youngwookim/data/perspective/{}_claim/q_res_100".format(split))
    candidate = get_candidates(q_res_path, split, config3())
    save_to_pickle(candidate, "pc_qk100_{}".format(split))



if __name__ == "__main__":
    run_get_for_token_scoring()
