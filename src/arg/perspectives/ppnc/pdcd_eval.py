import os
from typing import Dict, Tuple

import scipy.special

from arg.perspectives.eval_caches import eval_map
from arg.perspectives.ppnc.collect_score import load_combine_info_jsons
from arg.perspectives.types import DataID, CPIDPair
from cpath import output_path
from list_lib import lmap
from misc_lib import group_by
from tlm.estimator_prediction_viewer import EstimatorPredictionViewer


def get_confidence_or_rel_score(entry):
    try:
        return entry.get_vector("confidence")
    except KeyError:
        return entry.get_vector('rel_score')


def collect_scores_and_confidence(prediction_file, info: Dict, logit_to_score) \
        -> Dict[DataID, Tuple[CPIDPair, float, float]]:
    data = EstimatorPredictionViewer(prediction_file)
    print("Num data ", data.data_len)
    out_d: Dict[DataID, Tuple[CPIDPair, float, float]] = {}
    for entry in data:
        logits = entry.get_vector("logits")
        score = logit_to_score(logits)
        data_id = entry.get_vector("data_id")[0]
        confidence = get_confidence_or_rel_score(entry)
        try:
            cur_info = info[str(data_id)]
            cid = cur_info['cid']
            pid = cur_info['pid']
            cpid = CPIDPair((cid, pid))
            out_d[data_id] = (cpid, score, confidence)
        except KeyError as e:
            print("Key error")
            print("data_id", data_id)
            pass
    return out_d


def summarize_score(info_dir, prediction_file) -> Dict[CPIDPair, float]:
    info = load_combine_info_jsons(info_dir)

    def logit_to_score_softmax(logit):
        return scipy.special.softmax(logit)[1]

    scores: Dict[DataID, Tuple[CPIDPair, float, float]] = collect_scores_and_confidence(prediction_file, info, logit_to_score_softmax)
    grouped = group_by(scores.values(), lambda x: x[0])
    print("Group size:", len(grouped))
    out_d = {}
    for cpid, items in grouped.items():
        a = 0
        b = 0
        score_list = list([e[1] for e in items])
        conf_list = list([e[2] for e in items])

        print(cpid)
        print(lmap("{0:.2f}".format, score_list))
        print(lmap("{0:.2f}".format, conf_list))
        for cpid, score, confidence in items:
            a += score * confidence
            b += confidence

        final_score = a / (b + 0.0001)
        out_d[cpid] = final_score
    return out_d


def main():
    info_dir = "/mnt/nfs/work3/youngwookim/job_man/pdcd_val_info"
    prediction_file = os.path.join(output_path, "pdcd5.score")
    score_d = summarize_score(info_dir, prediction_file)
    map_score = eval_map("train", score_d)
    print(map_score)



if __name__ == "__main__":
    main()

