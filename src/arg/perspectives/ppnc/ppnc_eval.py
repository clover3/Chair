import os
from typing import Dict, Tuple

import scipy.special

from arg.perspectives.eval_caches import eval_map
from arg.perspectives.ppnc import collect_score
from arg.perspectives.ppnc.collect_score import load_combine_info_jsons
from arg.perspectives.types import DataID, CPIDPair
from cpath import output_path
from list_lib import right
from misc_lib import group_by, average


def top_k_average(items):
    k = 3
    items.sort(reverse=True)
    return average(items[:k])


def summarize_score(info_dir, prediction_file) -> Dict[CPIDPair, float]:
    info = load_combine_info_jsons(info_dir)
    def logit_to_score_reg(logit):
        return logit[0]

    def logit_to_score_softmax(logit):
        return scipy.special.softmax(logit)[1]

    scores: Dict[DataID, Tuple[CPIDPair, float]] = collect_score.collect_scores(prediction_file, info, logit_to_score_softmax)
    grouped = group_by(scores.values(), lambda x: x[0])
    print("Group size:", len(grouped))
    out_d = {}
    for cpid, items in grouped.items():
        final_score = top_k_average(right(items))
        out_d[cpid] = final_score
    return out_d


def main():
    info_dir = "/mnt/nfs/work3/youngwookim/job_man/ppnc_50_pers_val"
    prediction_file = os.path.join(output_path, "ppnc_50_val_prediction")
    score_d = summarize_score(info_dir, prediction_file)
    map_score = eval_map("train", score_d)
    print(map_score)


def debug():
    info_dir = "/mnt/nfs/work3/youngwookim/job_man/ppnc_50_pers_val"
    info = load_combine_info_jsons(info_dir)


if __name__ == "__main__":
    main()

