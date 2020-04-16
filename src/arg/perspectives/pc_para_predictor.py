import os
import pickle
from collections import Counter
from typing import Dict, List, Tuple

from arg.perspectives.collection_based_classifier import predict_interface
from arg.perspectives.cpid_def import CPID
from arg.perspectives.pc_para_eval import get_cpid_score
from base_type import FileName, FilePath
from cache import load_cache, save_to_pickle
from cpath import output_path, pjoin
from misc_lib import SuccessCounter


def get_cpid_score_from_cache_or_raw(pred_path, cpid_resolute, strategy):
    cache_name = os.path.basename(pred_path) + "_" + strategy

    r = load_cache(cache_name)
    if r is None:
        r = get_cpid_score(pred_path, cpid_resolute, strategy)

    save_to_pickle(r, cache_name)
    return r


def predict_by_para_scorer(score_pred_file_name: FileName,
                           cpid_resolute_file: FileName,
                           claims,
                           top_k) -> List[Tuple[str, List[Dict]]]:
    suc_count = SuccessCounter()
    suc_count.reset()

    pred_path: FilePath = pjoin(output_path, score_pred_file_name)
    print("Loading cpid_resolute")
    cpid_resolute: Dict[str, CPID] = load_cpid_resolute(cpid_resolute_file)
    print("Loading paragraph triple scores")
    score_d: Dict[CPID, float] = get_cpid_score_from_cache_or_raw(pred_path, cpid_resolute, "avg")

    per_claim_suc = {}
    per_claim_counter = {}

    def scorer(lucene_score, query_id):
        claim_id, p_id = query_id.split("_")
        if claim_id not in per_claim_suc:
            per_claim_counter[claim_id] = Counter()
            per_claim_suc[claim_id] = SuccessCounter()

        if query_id in score_d:
            cls_score = score_d[query_id]
            per_claim_suc[claim_id].suc()
            if cls_score > 0.8:
                per_claim_counter[claim_id][1] += 1
            elif cls_score < 0.3:
                per_claim_counter[claim_id][0] += 1
            suc_count.suc()
        else:
            cls_score = 0.5
            per_claim_suc[claim_id].fail()
            suc_count.fail()

        score = 0.9 * cls_score + 0.1 * lucene_score / 20
        return score

    r = predict_interface(claims, top_k, scorer)
    for claim in per_claim_suc:
        suc_counter = per_claim_suc[claim]
        print("{} suc/total={}/{}  True/False={}/{}".format(
            claim, suc_counter.get_suc(), suc_counter.get_total(),
            per_claim_counter[claim][1], per_claim_counter[claim][0]
        ))

    print("{} found of {}".format(suc_count.get_suc(), suc_count.get_total()))
    return r


def load_cpid_resolute(name: FileName):
    cpid_resolute: Dict[str, CPID] = pickle.load(open(pjoin(output_path, name), "rb"))
    return cpid_resolute
