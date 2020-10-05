

from typing import List

#
from arg.perspectives.load import load_train_claim_ids, get_claims_from_ids
from arg.perspectives.ppnc.resource import load_qk_candidate_train
from arg.qck.decl import QKUnit
from arg.qck.dynamic_kdp.requester import request_kdp_eval
from misc_lib import split_7_3


def main():
    print("Loading data ....")
    d_ids: List[int] = list(load_train_claim_ids())
    claims = get_claims_from_ids(d_ids)
    train, val = split_7_3(claims)

    val_cids = {str(t['cId']) for t in val}

    qk_candidate: List[QKUnit] = load_qk_candidate_train()
    qk_candidate_val = list([qk for qk in qk_candidate if qk[0].query_id in val_cids])

    print(qk_candidate_val[0][0])

    for q, kdp_list in qk_candidate_val[1:9]:
        job_id = request_kdp_eval(kdp_list)
        print('qid:', q.query_id)
        print('job_id', job_id)


if __name__ == "__main__":
    main()