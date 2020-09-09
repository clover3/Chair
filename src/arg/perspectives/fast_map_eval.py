import os
from typing import Dict

from arg.perspectives.cpid_def import CPID
from arg.perspectives.eval_caches import eval_map, get_ap_list_from_score_d, \
    get_acc_from_score_d
from arg.perspectives.ppnc.ppnc_eval import summarize_score
from arg.perspectives.types import CPIDPair
from cache import load_from_pickle
from cpath import output_path
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
    #target_cids = [628, 286, 591, 664, 842, 598, 707, 457, 166, 864, 493, 807, 609, 515, 641, 116, 496, 608, 24, 694, 684, 722, 572, 676, 160, 575, 514, 960, 927, 463, 838, 921, 638, 34, 835, 194, 464, 159, 595, 812, 25, 1004]

    #selected_score_d = {k: v for k, v in score_d.items() if k[0] in target_cids}
    #print("selected_score_d :", len(selected_score_d))

    map_score = eval_map("dev", score_d, True)
    print(map_score)


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
    bert_eval_acc()