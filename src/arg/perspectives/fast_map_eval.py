import os
from typing import Dict

from arg.perspectives.cpid_def import CPID
from arg.perspectives.eval_caches import eval_map, get_ap_list_from_score_d, \
    get_acc_from_score_d
from arg.perspectives.ppnc.ppnc_eval import summarize_score
from arg.perspectives.trec_helper import scrore_d_to_trec_style_predictions
from arg.perspectives.types import CPIDPair
from cache import load_from_pickle
from cpath import output_path
from trec.trec_parse import write_trec_ranked_list_entry
from list_lib import left
from tab_print import print_table


def ppnc_50_eval():
    info_dir = "/mnt/nfs/work3/youngwookim/job_man/ppnc_50_pers_val"
    prediction_file = os.path.join(output_path, "ppnc_50_val_prediction")
    score_d = summarize_score(info_dir, prediction_file)
    map_score = eval_map("train", score_d)
    print(map_score)

def CPID_to_CPIDPair(cpid: CPID) -> CPIDPair:
    cid, pid = cpid.split("_")
    return CPIDPair((int(cid), int(pid)))


def bert_eval():
    pc_score_d: Dict[CPID, float] = load_from_pickle("pc_bert_baseline_score_d")
    score_d: Dict[CPIDPair, float] = {CPID_to_CPIDPair(k): v for k, v in pc_score_d.items()}
    target_cids = [628, 286, 591, 664, 842, 598, 707, 457, 166, 864, 493, 807, 609, 515, 641, 116, 496, 608, 24, 694, 684, 722, 572, 676, 160, 575, 514, 960, 927, 463, 838, 921, 638, 34, 835, 194, 464, 159, 595, 812, 25, 1004]

    selected_score_d = {k: v for k, v in score_d.items() if k[0] in target_cids}
    print("selected_score_d :", len(selected_score_d))

    map_score = eval_map("dev", score_d, True)
    print(map_score)


def sanity_check():
    pc_score_d1: Dict[CPID, float] = load_from_pickle("pc_bert_baseline_score_d")
    pc_score_d1: Dict[CPIDPair, float] = {CPID_to_CPIDPair(k): v for k, v in pc_score_d1.items()}

    pc_score_d2: Dict[CPID, float] = load_from_pickle("stage2_score_d")
    run1_cids = set(left(pc_score_d1.keys()))
    run2_cids = set(left(pc_score_d2.keys()))
    assert all(cid in run1_cids for cid in run2_cids)

    run1_keys = set(pc_score_d1.keys())
    run1_keys_overlap = list([k for k in run1_keys if k[0] in run2_cids])
    run2_keys = set(pc_score_d1.keys())
    print(len(run1_keys))
    print(len(run2_keys))

    for key in run1_keys_overlap:
        assert key in run2_keys




def save_to_trec_format():
    pc_score_d: Dict[CPID, float] = load_from_pickle("pc_bert_baseline_score_d")
    run_name = "bert_baseline"
    score_d: Dict[CPIDPair, float] = {CPID_to_CPIDPair(k): v for k, v in pc_score_d.items()}
    save_to_common_path(run_name, score_d)


def save_to_trec_format2():
    pc_score_d_fail_back: Dict[CPID, float] = load_from_pickle("pc_bert_baseline_score_d")
    fail_back_score: Dict[CPIDPair, float] = {CPID_to_CPIDPair(k): v for k, v in pc_score_d_fail_back.items()}
    pc_score_d: Dict[CPIDPair, float] = load_from_pickle("stage2_score_d")

    for key in fail_back_score:
        if key not in pc_score_d:
            pc_score_d[key] = fail_back_score[key]

    run_name = "stage2_1_inc_failback"
    save_to_common_path(run_name, pc_score_d)


def save_to_common_path(run_name, score_d):
    ranked_list = scrore_d_to_trec_style_predictions(score_d, run_name)
    save_path = os.path.join(output_path, "perspective_ranked_list", run_name + ".txt")
    write_trec_ranked_list_entry(ranked_list, save_path)


def bert_eval_acc():
    pc_score_d: Dict[CPID, float] = load_from_pickle("pc_bert_baseline_score_d")
    score_d: Dict[CPIDPair, float] = {CPID_to_CPIDPair(k): v for k, v in pc_score_d.items()}
    #target_cids = [628, 286, 591, 664, 842, 598, 707, 457, 166, 864, 493, 807, 609, 515, 641, 116, 496, 608, 24, 694, 684, 722, 572, 676, 160, 575, 514, 960, 927, 463, 838, 921, 638, 34, 835, 194, 464, 159, 595, 812, 25, 1004]
    #selected_score_d = {k: v for k, v in score_d.items() if k[0] in target_cids}
    #print("selected_score_d :", len(selected_score_d))

    acc = get_acc_from_score_d(score_d, "dev")
    print(acc)


def bert_eval_all():
    pc_score_d: Dict[CPID, float] = load_from_pickle("pc_bert_baseline_score_d")
    score_d: Dict[CPIDPair, float] = {CPID_to_CPIDPair(k): v for k, v in pc_score_d.items()}
    ap_list, cids = get_ap_list_from_score_d(score_d, "dev")
    print_table(zip(cids, ap_list))


def main():
    ppnc_50_eval()


if __name__ == "__main__":
    save_to_trec_format2()
