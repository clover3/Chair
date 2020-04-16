from typing import List

from arg.perspectives.evaluate import evaluate
from arg.perspectives.load import get_claims_from_ids, load_dev_claim_ids
from arg.perspectives.pc_para_predictor import predict_by_para_scorer
from base_type import FileName


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
    #pred = predict_with_lm(val, top_k)
    print(evaluate(pred))


if __name__ == "__main__":
    run_baseline()
