import os
from typing import List, Dict, Tuple

from scipy import stats

from arg.perspectives.cpid_def import CPID
from arg.perspectives.eval_caches import get_joined_correctness
from arg.perspectives.fast_map_eval import CPID_to_CPIDPair
from arg.perspectives.ppnc.ppnc_eval import summarize_score
from arg.perspectives.types import CPIDPair
from cache import load_from_pickle
from cpath import output_path
from misc_lib import average, exist_or_mkdir


def get_bert_decisions() -> List[Tuple[int, List[int]]]:
    pc_score_d: Dict[CPID, float] = load_from_pickle("pc_bert_baseline_score_d")
    score_d: Dict[CPIDPair, float] = {CPID_to_CPIDPair(k): v for k, v in pc_score_d.items()}
    return get_joined_correctness(score_d, "dev")


def get_cppnc_decision():
    out_dir = os.path.join(output_path, "cppnc")
    exist_or_mkdir(out_dir)
    save_name = "cpnc4_triple_2"
    info_file_path = os.path.join(out_dir, save_name + ".info")
    pred_file_path = os.path.join(out_dir, save_name + ".score")
    score_d = summarize_score(info_file_path, pred_file_path)
    return get_joined_correctness(score_d, "dev")


def main():
    a_decisions = get_cppnc_decision()
    b_decisions = get_bert_decisions()
    b_decisions_d = dict(b_decisions)

    a_all = []
    b_all = []
    for cid, correctness_a in a_decisions:
        correctness_b = b_decisions_d[cid]
        assert len(correctness_a) == len(correctness_b)
        a_all.extend(correctness_a)
        b_all.extend(correctness_b)
    print(average(a_all))
    print(average(b_all))
    r = stats.ttest_ind(a_all, b_all)
    print(r)





if __name__ == "__main__":
    main()