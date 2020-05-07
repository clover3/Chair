from typing import List

from arg.perspectives.evaluate import evaluate
from arg.perspectives.load import get_claims_from_ids, load_dev_claim_ids
from arg.perspectives.pc_para_predictor import predict_by_para_scorer
from arg.perspectives.relevance_based_predictor import predict_from_dict, predict_from_two_dict
from arg.perspectives.runner.eval_cmd import eval_from_score_d
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



def run_eval_with_dict(pickle_name):
    d_ids: List[int] = list(load_dev_claim_ids())
    claims = get_claims_from_ids(d_ids)
    print("targets", len(claims))
    top_k = 5
    pc_score_d = load_from_pickle(pickle_name)
    pred = predict_from_dict(pc_score_d, claims, top_k)
    print(evaluate(pred))


def run_eval_with_two_dict():
    d_ids: List[int] = list(load_dev_claim_ids())
    claims = get_claims_from_ids(d_ids)
    print("targets", len(claims))
    top_k = 5
    pc_score_d = load_from_pickle("pc_bert_baseline_score_d")
    pc_score_d2 = load_from_pickle("pc_rel_based_score_dev")
    pred = predict_from_two_dict(pc_score_d, pc_score_d2, claims, top_k)
    print(evaluate(pred))


def run_rel_based():
    run_eval_with_dict("pc_rel_based_score_dev")


def run_rel_filter_eval():
    run_eval_with_dict("tf_rel_filter_B_dev_score")


def run_concat_rep_eval():
    run_eval_with_dict("pc_concat_dev_score")


def run_bert_baseline():
    run_eval_with_dict("pc_bert_baseline_score_d")


def run_bert_baseline2():
    score_d = load_from_pickle("pc_bert_baseline_score_d2")
    print(eval_from_score_d(score_d, 5))

    run_eval_with_dict("pc_bert_baseline_score_d2")


if __name__ == "__main__":
    run_bert_baseline()
