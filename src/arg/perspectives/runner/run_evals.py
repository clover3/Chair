import os
from typing import Dict, List, Set, Tuple

from arg.perspectives.basic_analysis import predict_by_elastic_search, predict_by_oracle_on_candidate
from arg.perspectives.bm25_predict import predict_by_bm25, get_bm25_module, predict_by_bm25_from_candidate
from arg.perspectives.claim_lm.lm_predict import predict_by_lm
from arg.perspectives.claim_lm.passage_to_lm import get_train_passage_a_lms
from arg.perspectives.cpid_def import CPID
from arg.perspectives.eval_helper import claim_as_query, get_eval_candidates_l, \
    prediction_to_trec_format, get_eval_candidates_w_q_text
from arg.perspectives.evaluate import evaluate, evaluate_map, evaluate_recall
from arg.perspectives.load import load_train_claim_ids, get_claims_from_ids
from arg.perspectives.pc_para_predictor import load_cpid_resolute, predict_by_para_scorer
from arg.perspectives.query.load_rm import get_expanded_query_text
from arg.perspectives.relevance_based_predictor import predict_from_dict
from arg.perspectives.reweight_predict import predict_by_reweighter
from arg.perspectives.runner_uni.build_topic_lm import build_gold_claim_lm_train, build_baseline_lms, ClaimLM
from arg.perspectives.split_helper import train_split
from base_type import FileName
from cache import load_from_pickle
from cpath import output_path
from evals.trec import write_trec_ranked_list_entry
from list_lib import lmap, lfilter


def filter_avail(claims):
    cpid_resolute: Dict[str, CPID] = load_cpid_resolute(FileName("resolute_dict_580_606"))
    cid_list: List[int] = lmap(lambda x: int(x.split("_")[0]), cpid_resolute.values())
    cid_list: Set[int] = set(cid_list)
    return lfilter(lambda x: x['cId'] in cid_list, claims)


def run_para_scorer():
    claims, val = train_split()
    top_k = 6

    target = filter_avail(val)
    print("targets", len(target))
    score_pred_file: FileName = FileName("pc_para_D_pred")
    cpid_resolute_file: FileName = FileName("resolute_dict_580_606")
    pred = predict_by_para_scorer(score_pred_file, cpid_resolute_file,
                                  target, top_k)
    print(evaluate(pred))


def run_rel_scorer():
    claims, val = train_split()
    top_k = 6
    target = filter_avail(val)
    print("targets", len(target))
    pc_score_d = load_from_pickle("pc_rel_based_score_train")
    pred = predict_from_dict(pc_score_d, target, top_k)
    print(evaluate(pred))


def run_bert_baseline():
    claims, val = train_split()
    top_k = 50
    target = filter_avail(val)
    print("targets", len(target))
    pc_score_d = load_from_pickle("pc_bert_baseline_score_d_train")
    pred = predict_from_dict(pc_score_d, target, top_k)
    print(evaluate(pred))


def run_baseline():
    claims, val = train_split()
    top_k = 50
    pred = predict_by_elastic_search(claims, top_k)
    print(evaluate(pred))


def run_oracle_on_candiate():
    claims, val = train_split()
    top_k = 5
    pred = predict_by_oracle_on_candidate(claims, top_k)
    print(evaluate(pred))

def run_oracle_on_candiate_map():
    claims, val = train_split()
    top_k = 50
    pred = predict_by_oracle_on_candidate(claims, top_k)
    print(evaluate_map(pred))


def run_bm25():
    claims, val = train_split()
    top_k = 20
    pred = predict_by_bm25(get_bm25_module(), claims, top_k)
    print(evaluate(pred))



def run_gold_lm():
    claims, val = train_split()
    top_k = 5
    print("Building lms")
    claim_lms: List[ClaimLM] = build_gold_claim_lm_train()
    print("Predicting")
    pred = predict_by_lm(claim_lms, claims, top_k)
    print(evaluate(pred))


def run_baseline_lm():
    claims, val = train_split()
    claims = val
    top_k = 50
    print("Building lms")
    claim_lms = build_baseline_lms(claims)
    print("Predicting")
    pred = predict_by_lm(claim_lms, claims, top_k)
    print(evaluate_map(pred))


def run_a_relevant_lm():
    claims, val = train_split()
    top_k = 50
    print("Building lms")
    claim_lms = get_train_passage_a_lms()
    print("Predicting")
    pred = predict_by_lm(claim_lms, claims, top_k)
    print(evaluate_map(pred))


def run_bm25_map():
    claims, val = train_split()
    top_k = 50
    pred = predict_by_bm25(get_bm25_module(), claims, top_k)
    print(evaluate_map(pred))


def run_gold_lm_ap():
    claims, val = train_split()
    top_k = 50
    print("Building lms")
    claim_lms = build_gold_claim_lm_train()
    print("Predicting")
    pred = predict_by_lm(claim_lms, claims, top_k)
    print(evaluate_map(pred))


def run_reweight():
    top_k = 7
    claims, val = train_split()
    param = {'k1': 1}
    target = claims[:50]
    pred = predict_by_reweighter(get_bm25_module(), target, top_k, param)
    print(param)
    print(evaluate(pred))


def run_bm25_2():
    claims, val = train_split()
    top_k = 1000
    candidate_dict: List[Tuple[int, List[int]]] = get_eval_candidates_w_q_text(claim_as_query(claims), top_k)
    pred = predict_by_bm25_from_candidate(get_bm25_module(), claims, candidate_dict, top_k)
    print(evaluate_recall(pred, True))


def save_bm25_as_trec_format():
    d_ids: List[int] = list(load_train_claim_ids())
    claims = get_claims_from_ids(d_ids)
    top_k = 200
    candidate_dict: List[Tuple[int, List[int]]] = get_eval_candidates_w_q_text(claim_as_query(claims), top_k)
    pred = predict_by_bm25_from_candidate(get_bm25_module(), claims, candidate_dict, top_k)
    entries = prediction_to_trec_format(pred, "bm25")
    write_trec_ranked_list_entry(entries, os.path.join(output_path, "ranked_list", "bm25.txt"))


def run_bm25_ex():
    claims, val = train_split()
    top_k = 100
    candidate_dict = get_eval_candidates_l(get_expanded_query_text(claims, "train"))
    pred = predict_by_bm25_from_candidate(get_bm25_module(), claims, candidate_dict, top_k)
    print(evaluate_recall(pred, True))


if __name__ == "__main__":
    run_bert_baseline()
