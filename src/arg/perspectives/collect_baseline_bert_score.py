import os
from typing import List, Dict

from scipy.special import softmax

from arg.perspectives.basic_analysis import load_data_point
from arg.perspectives.cpid_def import CPID
from arg.perspectives.declaration import PerspectiveCandidate
from cache import save_to_pickle
from cpath import output_path
from tlm.estimator_prediction_viewer import EstimatorPredictionViewer


def collect_by_order(input_file, feature_data: List[PerspectiveCandidate]):
    predictions = EstimatorPredictionViewer(input_file)

    print("prediction : {}".format(predictions.data_len))
    print("feature_data : {}".format(len(feature_data)))

    score_d: Dict[CPID, float] = {}
    for pred_entry, pc_candidate in zip(predictions, feature_data):
        logits = pred_entry.get_vector("logits")
        probs = softmax(logits)
        score = probs[1]

        cpid = CPID("{}_{}".format(pc_candidate.cid, pc_candidate.pid))
        score_d[cpid] = score

    return score_d


def collect_bert():
    split = "dev"
    data: List[PerspectiveCandidate] = load_data_point(split)
    pred_path = os.path.join(output_path, "pc_bert_baseline2")
    score_d = collect_by_order(pred_path, data)
    save_to_pickle(score_d, "pc_bert_baseline_score_d2")

    # split = "train"
    # data: List[PerspectiveCandidate] = load_data_point(split)
    # pred_path = os.path.join(output_path, "pc_bert_baseline_train")
    # score_d = collect(pred_path, data)
    # save_to_pickle(score_d, "pc_bert_baseline_score_d_train")

def collect_aux():
    split = "dev"
    data: List[PerspectiveCandidate] = load_data_point(split)
    pred_path = os.path.join(output_path, "pc_aux")
    score_d = collect_by_order(pred_path, data)
    save_to_pickle(score_d, "pc_aux_score_d")

if __name__ == "__main__":
    collect_aux()
