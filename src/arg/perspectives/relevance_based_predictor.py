from collections import Counter
from typing import Dict, List, Tuple

from arg.perspectives.collection_based_classifier import predict_interface
from arg.perspectives.cpid_def import CPID
from misc_lib import SuccessCounter


def predict_from_dict(score_d: Dict[CPID, float],
                      claims,
                      top_k) -> List[Tuple[str, List[Dict]]]:
    suc_count = SuccessCounter()
    suc_count.reset()

    per_claim_suc = {}
    per_claim_counter = {}

    rationale_d = {}

    def scorer(lucene_score, query_id):
        claim_id, p_id = query_id.split("_")
        if claim_id not in per_claim_suc:
            per_claim_counter[claim_id] = Counter()
            per_claim_suc[claim_id] = SuccessCounter()

        cls_score = get_score_by_d(claim_id, query_id)

        score = (cls_score < 4) * -1 + lucene_score / 20
        #score = cls_score + lucene_score / 20
        score = cls_score
        r = "score={0:.2f} <- cls_score({1:.2f}) lucene_score({2:.2f}) /20".format(score, cls_score, lucene_score)
        rationale_d[query_id] = r
        return score

    def get_score_by_d(claim_id, query_id):
        if query_id in score_d:
            cls_score = score_d[query_id]
            per_claim_suc[claim_id].suc()
            if cls_score > 0.8:
                per_claim_counter[claim_id][1] += 1
            elif cls_score < 0.3:
                per_claim_counter[claim_id][0] += 1
            suc_count.suc()
        else:
            cls_score = 0
            per_claim_suc[claim_id].fail()
            suc_count.fail()
        return cls_score

    def get_rationale(query_id):
        if query_id in rationale_d:
            return rationale_d[query_id]
        else:
            return "(N/A)"

    r = predict_interface(claims, top_k, scorer, get_rationale)
    for claim in per_claim_suc:
        suc_counter = per_claim_suc[claim]
        print("{} suc/total={}/{}  True/False={}/{}".format(
            claim, suc_counter.get_suc(), suc_counter.get_total(),
            per_claim_counter[claim][1], per_claim_counter[claim][0]
        ))

    print("{} found of {}".format(suc_count.get_suc(), suc_count.get_total()))
    return r


def predict_from_two_dict(score_d: Dict[CPID, float],
                          score_d2: Dict[CPID, float],
                          claims,
                          top_k) -> List[Tuple[str, List[Dict]]]:
    suc_count = SuccessCounter()
    suc_count.reset()

    per_claim_suc = {}
    per_claim_counter = {}

    rationale_d = {}

    def scorer(lucene_score, query_id):
        claim_id, p_id = query_id.split("_")
        if claim_id not in per_claim_suc:
            per_claim_counter[claim_id] = Counter()
            per_claim_suc[claim_id] = SuccessCounter()

        cls_score = get_score_by_d(claim_id, query_id)
        cls_score2 = get_score_by_d2(claim_id, query_id)

        score = cls_score * 100 + cls_score2
        r = "score={0:.2f} <- cls_score({1:.2f}/{2:.2f}) lucene_score({3:.2f}) /20".format(score,
                                                                                           cls_score,
                                                                                           cls_score2,
                                                                                           lucene_score)
        rationale_d[query_id] = r
        return score

    def get_score_by_d(claim_id, query_id):
        if query_id in score_d:
            cls_score = score_d[query_id]
            per_claim_suc[claim_id].suc()
            if cls_score > 0.8:
                per_claim_counter[claim_id][1] += 1
            elif cls_score < 0.3:
                per_claim_counter[claim_id][0] += 1
            suc_count.suc()
        else:
            cls_score = 0
            per_claim_suc[claim_id].fail()
            suc_count.fail()
        return cls_score

    def get_score_by_d2(claim_id, query_id):
        if query_id in score_d2:
            cls_score = score_d2[query_id]
            per_claim_suc[claim_id].suc()
            if cls_score > 0.8:
                per_claim_counter[claim_id][1] += 1
            elif cls_score < 0.3:
                per_claim_counter[claim_id][0] += 1
            suc_count.suc()
        else:
            cls_score = 0
            per_claim_suc[claim_id].fail()
            suc_count.fail()
        return cls_score

    def get_rationale(query_id):
        if query_id in rationale_d:
            return rationale_d[query_id]
        else:
            return "(N/A)"

    r = predict_interface(claims, top_k, scorer, get_rationale)
    for claim in per_claim_suc:
        suc_counter = per_claim_suc[claim]
        print("{} suc/total={}/{}  True/False={}/{}".format(
            claim, suc_counter.get_suc(), suc_counter.get_total(),
            per_claim_counter[claim][1], per_claim_counter[claim][0]
        ))

    print("{} found of {}".format(suc_count.get_suc(), suc_count.get_total()))
    return r


def prediction_to_dict(prediction: List[Tuple[str, List[Dict]]]) -> Dict[CPID, float]:
    output: Dict[CPID,float] = {}
    for claim_id, preds in prediction:

        for pred in preds:
            cpid = CPID("{}_{}".format(claim_id, pred['pid']))
            score = pred['score']
            output[cpid] = float(score)

    return output







