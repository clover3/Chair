import os
import pickle
from typing import Dict, List, Tuple

from arg.perspectives.collection_based_classifier import predict_interface
from arg.perspectives.cpid_def import CPID
from arg.perspectives.pc_para_eval import get_cpid_score
from cpath import output_path
from misc_lib import SuccessCounter


def predict_by_para_scorer(claims, top_k) -> List[Tuple[str, List[Dict]]]:
    suc_count = SuccessCounter()
    suc_count.reset()

    pred_path = os.path.join(output_path, "pc_para_D_pred")
    cpid_resolute: Dict[str, CPID] = load_cpid_resolute()

    score_d: Dict[CPID, float] = get_cpid_score(pred_path, cpid_resolute, "avg")

    def scorer(lucene_score, query_id):
        if query_id in score_d:
            score = score_d[query_id]
            suc_count.suc()
        else:
            score = -1
            suc_count.fail()
        return score

    r = predict_interface(claims, top_k, scorer)
    print("{} found of {}".format(suc_count.suc(), suc_count.fail()))
    return r


def load_cpid_resolute():
    cpid_resolute: Dict[str, CPID] = pickle.load(open(os.path.join(output_path, "resolute_dict_580_606"), "rb"))
    return cpid_resolute
