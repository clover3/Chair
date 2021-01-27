from collections import Counter
from typing import Dict

from arg.perspectives.load import get_claims_from_ids
from arg.perspectives.new_split.common import get_qids_for_split, split_name2
from arg.perspectives.new_split.qk_common import load_all_qk
from arg.perspectives.runner_uni.build_topic_lm import build_gold_lms
from arg.qck.filter_qk import filter_qk_rel
from cache import save_to_pickle
from list_lib import lmap


def get_claim_lms() -> Dict[str, Counter]:
    split = "train"
    qids = list(get_qids_for_split(split_name2, split))
    cids = lmap(int, qids)
    claims = get_claims_from_ids(cids)
    claim_lms = build_gold_lms(claims)
    claim_lms_dict: Dict[str, Counter] = {str(claim_lm.cid): claim_lm.LM for claim_lm in claim_lms}
    return claim_lms_dict


def main():
    split = "train"
    all_qk = load_all_qk()
    qids = list(get_qids_for_split(split_name2, split))
    qks_for_split = list([qk for qk in all_qk if qk[0].query_id in qids])
    query_lms: Dict[str, Counter] = get_claim_lms()
    print(len(qks_for_split), len(query_lms))
    filtered_qk_candidate = filter_qk_rel(qks_for_split, query_lms, 50)
    save_to_pickle(filtered_qk_candidate, "pc_qk3_filtered_rel_{}".format(split))


if __name__ == "__main__":
    main()
