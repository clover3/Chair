import os
import sys
from typing import List, Dict, Tuple

import scipy.special

from arg.perspectives.ppnc.pdcd_eval import collect_scores_and_confidence
from arg.perspectives.types import DataID, CPIDPair
from cpath import output_path
from list_lib import dict_value_map, lmap
from misc_lib import group_by, SuccessCounter, exist_or_mkdir
from tlm.estimator_output_reader import load_combine_info_jsons
from visualize.html_visual import Cell, HtmlVisualizer


def get_confidence_list_per_cid(info_dir, prediction_file) -> Dict[int, List[float]]:
    info = load_combine_info_jsons(info_dir)

    def logit_to_score_softmax(logit):
        return scipy.special.softmax(logit)[1]

    scores: Dict[DataID, Tuple[CPIDPair, float, float]] = collect_scores_and_confidence(prediction_file, info, logit_to_score_softmax)
    grouped = group_by(scores.values(), lambda x: x[0])
    print("Group size:", len(grouped))
    entries = group_by_cpid(grouped)

    cid_grouped = group_by(entries, lambda x: x[0])
    verify_confidence_consistency(cid_grouped)

    return dict_value_map(lambda x: x[0][2], cid_grouped)


def verify_confidence_consistency(grouped):
    suc = SuccessCounter()
    for cid, items in grouped.items():
        confidence_list_list = list([confidence_list for cid_, pid, confidence_list in items])
        first_confidence_list = confidence_list_list[0]
        any_length = len(first_confidence_list)
        if any_length == 0:
            print("cid {} has no passages".format(cid))
            continue
        for l in confidence_list_list:
            if any_length != len(l):
                print(any_length, len(l))
                print(lmap(len, confidence_list_list))

        try:
            for other_confidence in confidence_list_list[1:]:
                for k in range(any_length):

                    if abs(first_confidence_list[k] - other_confidence[k]) > 0.1:
                        suc.fail()
                    else:
                        suc.suc()
        except IndexError:
            pass
    print("Confidence consistency : ", suc.get_suc_prob())


def group_by_cpid(grouped):
    entries = []
    for cpid, items in grouped.items():
        confidence_list = [e[2] for e in items]
        cid, pid = cpid
        entries.append((cid, pid, confidence_list))
    return entries


def main():
    save_name = sys.argv[1]
    out_dir = os.path.join(output_path, "cppnc")
    exist_or_mkdir(out_dir)
    info_file_path = os.path.join(out_dir, "cppnc_triple_all_dev_info")
    pred_file_path = os.path.join(out_dir, save_name + ".score")
    cid_and_confidences = get_confidence_list_per_cid(info_file_path, pred_file_path)

    rows = []
    for cid, confidenc_list in cid_and_confidences.items():
        row = list()
        row.append(Cell(str(cid)))
        row.extend([Cell("", highlight_score=c*100) for c in confidenc_list])
        rows.append(row)

    html = HtmlVisualizer("confidence.html")
    html.write_table(rows)


if __name__ == "__main__":
    main()

