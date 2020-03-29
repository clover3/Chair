from typing import Dict, List, Set

from arg.perspectives.cpid_def import CPID
from arg.perspectives.evaluate import evaluate
from arg.perspectives.load import get_claims_from_ids, load_train_claim_ids
from arg.perspectives.pc_para_predictor import load_cpid_resolute, predict_by_para_scorer
from base_type import FileName
from list_lib import lmap, lfilter
from misc_lib import split_7_3


def filter_avail(claims):
    cpid_resolute: Dict[str, CPID] = load_cpid_resolute(FileName("resolute_dict_580_606"))
    cid_list: List[int] = lmap(lambda x: int(x.split("_")[0]), cpid_resolute.values())
    cid_list: Set[int] = set(cid_list)
    return lfilter(lambda x: x['cId'] in cid_list, claims)


def run_baseline():
    d_ids: List[int] = list(load_train_claim_ids())
    claims = get_claims_from_ids(d_ids)
    train, val = split_7_3(claims)
    top_k = 5

    target = filter_avail(val)
    print("targets", len(target))
    #pred = predict_by_elastic_search(claims, top_k)
    #pred = predict_by_mention_num(target, top_k)
    score_pred_file: FileName = FileName("pc_para_D_pred")
    cpid_resolute_file: FileName = FileName("resolute_dict_580_606")
    pred = predict_by_para_scorer(score_pred_file, cpid_resolute_file,
                                  target, top_k)
    #pred = predict_with_lm(val, top_k)
    print(evaluate(pred))


if __name__ == "__main__":
    run_baseline()
