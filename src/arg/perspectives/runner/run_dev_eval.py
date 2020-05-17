from typing import List

from arg.perspectives.bm25_predict import predict_by_bm25, get_bm25_module, predict_by_bm25_rm
from arg.perspectives.evaluate import evaluate
from arg.perspectives.lm_predict import predict_by_lm, load_collection_tf
from arg.perspectives.load import get_claims_from_ids, load_dev_claim_ids
from arg.perspectives.pc_para_predictor import predict_by_para_scorer
from arg.perspectives.random_walk.pc_predict import pc_predict_from_vector_query
from arg.perspectives.relevance_based_predictor import predict_from_dict, predict_from_two_dict, prediction_to_dict
from arg.perspectives.runner.eval_cmd import eval_from_score_d
from base_type import FileName
from cache import load_from_pickle, save_to_pickle


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
    top_k = 8
    pc_score_d = load_from_pickle(pickle_name)
    pred = predict_from_dict(pc_score_d, claims, top_k)
    print(evaluate(pred))


def run_eval_with_two_dict():
    d_ids: List[int] = list(load_dev_claim_ids())
    claims = get_claims_from_ids(d_ids)
    print("targets", len(claims))
    top_k = 7
    pc_score_d = load_from_pickle("pc_bert_baseline_score_d")
    pc_score_d2 = load_from_pickle("pc_random_walk_based_score_d")
    pred = predict_from_two_dict(pc_score_d, pc_score_d2, claims, top_k)
    print(evaluate(pred))


def save_random_walk_pred():
    d_ids: List[int] = list(load_dev_claim_ids())
    claims = get_claims_from_ids(d_ids)
    top_k = 50
    q_tf_replace = dict(load_from_pickle("random_walk_score_100"))
    bm25 = get_bm25_module()
    pred = pc_predict_from_vector_query(bm25, q_tf_replace, claims, top_k)
    score_d = prediction_to_dict(pred)
    save_to_pickle(score_d, "pc_random_walk_based_score_d")


def run_bm25():
    d_ids: List[int] = list(load_dev_claim_ids())
    claims = get_claims_from_ids(d_ids)
    top_k = 7
    pred = predict_by_bm25(get_bm25_module(), claims, top_k)
    print(evaluate(pred))


def run_bm25_rm():
    d_ids: List[int] = list(load_dev_claim_ids())
    claims = get_claims_from_ids(d_ids)
    rm_info = load_from_pickle("perspective_dev_claim_rm")
    top_k = 7
    pred = predict_by_bm25_rm(get_bm25_module(), rm_info, claims, top_k)
    print(evaluate(pred))



def run_random_walk_score():
    d_ids: List[int] = list(load_dev_claim_ids())
    claims = get_claims_from_ids(d_ids)
    top_k = 7
    q_tf_replace = dict(load_from_pickle("random_walk_score_100"))
    #q_tf_replace = dict(load_from_pickle("pc_dev_par_tf"))
    #q_tf_replace = dict(load_from_pickle("bias_random_walk_dev_plus_all"))
    bm25 = get_bm25_module()
    pred = pc_predict_from_vector_query(bm25, q_tf_replace, claims, top_k)
    print(evaluate(pred))


def run_lm():
    d_ids: List[int] = list(load_dev_claim_ids())
    claims = get_claims_from_ids(d_ids)
    top_k = 7
    q_tf_replace = dict(load_from_pickle("pc_dev_par_tf"))
#    q_tf_replace = dict(load_from_pickle("random_walk_score_100"))
    bm25 = get_bm25_module()
    ctf = load_collection_tf()
    pred = predict_by_lm(q_tf_replace, ctf, bm25, claims, top_k)
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


def run_logit_baseline():
    run_eval_with_dict("pc_ngram_logits")


if __name__ == "__main__":
    run_eval_with_two_dict()
