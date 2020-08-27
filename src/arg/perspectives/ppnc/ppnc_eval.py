import os
from typing import List, Dict, Tuple, Set

from arg.perspectives.eval_helper import get_eval_candidates_from_pickle, predict_from_dict
from arg.perspectives.evaluate import evaluate_map
from arg.perspectives.ppnc import collect_score
from arg.perspectives.ppnc.collect_score import load_combine_info_jsons
from arg.perspectives.types import DataID, CPIDPair
from cpath import output_path
from list_lib import right, left, lfilter
from misc_lib import group_by


def summarize_score(info_dir, prediction_file) -> Dict[CPIDPair, float]:
    info = load_combine_info_jsons(info_dir)
    scores: Dict[DataID, Tuple[CPIDPair, float]] = collect_score.collect_scores(prediction_file, info)
    grouped = group_by(scores.values(), lambda x: x[0])
    print("Group size:", len(grouped))
    out_d = {}
    for cpid, items in grouped.items():
        final_score = max(right(items))
        out_d[cpid] = final_score
    return out_d


def eval_map(split, score_d: Dict[CPIDPair, float]):
    # load pre-computed perspectives
    candidates: List[Tuple[int, List[Dict]]] = get_eval_candidates_from_pickle(split)
    # only evalaute what's available
    valid_cids: Set[int] = set(left(score_d.keys()))
    sub_candidates: List[Tuple[int, List[Dict]]] = lfilter(lambda x: x[0] in valid_cids, candidates)
    print("{} claims are evaluated".format(len(sub_candidates)))
    predictions = predict_from_dict(score_d, sub_candidates, 50)
    return evaluate_map(predictions, True)


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

