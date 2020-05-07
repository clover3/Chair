import sys
from typing import List, Dict, Tuple

from arg.perspectives.cpid_def import CPID
from arg.perspectives.evaluate import evaluate
from arg.perspectives.pc_para_eval import get_cpid_score
from arg.perspectives.pc_para_predictor import load_cpid_resolute
from base_type import FileName
from cache import load_from_pickle
from list_lib import lmap


def load_dev_candiate() -> List[Tuple[Dict, List[Dict]]]:
    return load_from_pickle("pc_dev_candidate")


def eval_from_prediction(prediction_path):
    cpid_resolute_file: FileName = FileName("resolute_dict_dev_11")
    top_k = 5
    cpid_resolute: Dict[str, CPID] = load_cpid_resolute(cpid_resolute_file)

    print("cpid_resolute has {}".format(len(cpid_resolute)))
    strategy = "avg"
    score_d: Dict[CPID, float] = get_cpid_score(prediction_path, cpid_resolute, strategy)
    return eval_from_score_d(score_d, top_k)


def eval_from_score_d(score_d: Dict[CPID, float], top_k):
    candidate: List[Tuple[Dict, List[Dict]]] = load_dev_candiate()
    dp_not_found = 0
    def get_predictions(claim_and_candidate: Tuple[Dict, List[Dict]]) -> Tuple[str, List[Dict]]:
        claim_info, candidates = claim_and_candidate
        nonlocal dp_not_found
        for candi in candidates:
            cid = candi['cid']
            pid = candi['pid']
            cpid = CPID("{}_{}".format(cid, pid))

            if cpid in score_d:
                candi['new_score'] = score_d[cpid]
            else:
                dp_not_found += 1
                candi['new_score'] = 0.01

            candi['final_score'] = candi['new_score'] + candi['score'] / 100
            candi['rationale'] = "final_score={}  cls_score={}  lucene_score={}".format(
                candi['final_score'], candi['new_score'], candi['score']
            )

        candidates.sort(key=lambda c: c['final_score'], reverse=True)
        return claim_info['cId'], candidates[:top_k]
    predictions = lmap(get_predictions, candidate)
    print("{} data points are not found in predictions".format(dp_not_found))
    r = evaluate(predictions, debug=False)
    print(r)
    return r


if __name__ == "__main__":
    prediction_path = sys.argv[1]
    eval_from_prediction(prediction_path)
