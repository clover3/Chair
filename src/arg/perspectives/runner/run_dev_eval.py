from typing import List

from arg.perspectives.evaluate import evaluate
from arg.perspectives.load import get_claims_from_ids, load_dev_claim_ids
from arg.perspectives.pc_para_predictor import predict_by_para_scorer
from arg.perspectives.relevance_based_predictor import predict_from_dict
from base_type import FileName
from cache import load_from_pickle


def run_baseline():
    d_ids: List[int] = list(load_dev_claim_ids())
    claims = get_claims_from_ids(d_ids)
    print("targets", len(claims))
    top_k = 5
    score_pred_file: FileName = FileName("pc_para_D_pred_dev_11")
    cpid_resolute_file: FileName = FileName("resolute_dict_dev_11")
    # score_pred_file: FileName = FileName("pc_para_D_pred_dev")
    # cpid_resolute_file: FileName = FileName("resolute_dict_dev")
    pred = predict_by_para_scorer(score_pred_file,
                                  cpid_resolute_file,
                                  claims,
                                  top_k)
    print(evaluate(pred))


def run_rel_based():
    d_ids: List[int] = list(load_dev_claim_ids())
    claims = get_claims_from_ids(d_ids)
    print("targets", len(claims))
    top_k = 5
    pc_score_d = load_from_pickle("pc_rel_based_score_dev")
    pred = predict_from_dict(pc_score_d, claims, top_k)
    print(evaluate(pred))


def run_rel_filter_eval():
    d_ids: List[int] = list(load_dev_claim_ids())
    claims = get_claims_from_ids(d_ids)
    print("targets", len(claims))
    top_k = 5
    pc_score_d = load_from_pickle("tf_rel_filter_B_dev_score")
    pred = predict_from_dict(pc_score_d, claims, top_k)
    print(evaluate(pred))


if __name__ == "__main__":
    run_rel_filter_eval()
