from typing import Dict, List, Set

from arg.perspectives.basic_analysis import predict_by_elastic_search
from arg.perspectives.bm25_predict import predict_by_bm25, get_bm25_module
from arg.perspectives.cpid_def import CPID
from arg.perspectives.evaluate import evaluate
from arg.perspectives.load import get_claims_from_ids, load_train_claim_ids
from arg.perspectives.pc_para_predictor import load_cpid_resolute, predict_by_para_scorer
from arg.perspectives.relevance_based_predictor import predict_from_dict
from arg.perspectives.reweight_predict import predict_by_reweighter
from base_type import FileName
from cache import load_from_pickle
from list_lib import lmap, lfilter
from misc_lib import split_7_3


def filter_avail(claims):
    cpid_resolute: Dict[str, CPID] = load_cpid_resolute(FileName("resolute_dict_580_606"))
    cid_list: List[int] = lmap(lambda x: int(x.split("_")[0]), cpid_resolute.values())
    cid_list: Set[int] = set(cid_list)
    return lfilter(lambda x: x['cId'] in cid_list, claims)


def train_split():
    d_ids: List[int] = list(load_train_claim_ids())
    claims = get_claims_from_ids(d_ids)
    train, val = split_7_3(claims)
    return claims, val


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
    top_k = 6
    target = filter_avail(val)
    print("targets", len(target))
    pc_score_d = load_from_pickle("pc_bert_baseline_score_d_train")
    pred = predict_from_dict(pc_score_d, target, top_k)
    print(evaluate(pred))


def run_baseline():
    claims, val = train_split()
    top_k = 20
    pred = predict_by_elastic_search(claims, top_k)
    print(evaluate(pred))


def run_bm25():
    claims, val = train_split()
    top_k = 20
    pred = predict_by_bm25(get_bm25_module(), claims, top_k)
    print(evaluate(pred))


def run_reweight():
    top_k = 7
    claims, val = train_split()
    param = {'k1': 1}
    target = claims[:50]
    pred = predict_by_reweighter(get_bm25_module(), target, top_k, param)
    print(param)
    print(evaluate(pred))


if __name__ == "__main__":
    run_reweight()
