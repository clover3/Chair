import pickle
from typing import Dict, List, Tuple

from arg.perspectives.collection_based_classifier import predict_interface
from arg.perspectives.cpid_def import CPID
from arg.perspectives.pc_para_eval import get_cpid_score
from base_type import FileName, FilePath
from cpath import output_path, pjoin
from misc_lib import SuccessCounter


def predict_by_para_scorer(score_pred_file_name: FileName,
                           cpid_resolute_file: FileName,
                           claims,
                           top_k) -> List[Tuple[str, List[Dict]]]:
    suc_count = SuccessCounter()
    suc_count.reset()

    pred_path: FilePath = pjoin(output_path, score_pred_file_name)
    cpid_resolute: Dict[str, CPID] = load_cpid_resolute(cpid_resolute_file)
    score_d: Dict[CPID, float] = get_cpid_score(pred_path, cpid_resolute, "avg")
    print()
    print(score_d.keys())

    def scorer(lucene_score, query_id):
        if query_id in score_d:
            score = score_d[query_id]
            suc_count.suc()
        else:
            score = -1
            suc_count.fail()
        return score

    r = predict_interface(claims, top_k, scorer)
    print("{} found of {}".format(suc_count.get_suc(), suc_count.get_total()))
    return r


def load_cpid_resolute(name: FileName):
    cpid_resolute: Dict[str, CPID] = pickle.load(open(pjoin(output_path, name), "rb"))
    return cpid_resolute
