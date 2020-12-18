from collections import Counter
from typing import List, Dict

from arg.perspectives.ppnc.resource import load_qk_candidate_train, load_qk_candidate_dev
from arg.perspectives.runner_uni.build_topic_lm import build_gold_lms_for_split, ClaimLM
from arg.qck.filter_qk import filter_qk
from cache import save_to_pickle


def get_query_lms(split) -> Dict[str, Counter]:
    claim_lms: List[ClaimLM] = build_gold_lms_for_split(split)
    claim_lms_dict: Dict[str, Counter] = {str(claim_lm.cid): claim_lm.LM for claim_lm in claim_lms}
    return claim_lms_dict


def main():
    def get_qk_candidate(split):
        if split == "train":
            return load_qk_candidate_train()
        elif split == "dev":
            return load_qk_candidate_dev()

    for split in ["train", "dev"]:
        qk_candidate = get_qk_candidate(split)
        query_lms: Dict[str, Counter] = get_query_lms(split)
        print(len(qk_candidate), len(query_lms))
        filtered_qk_candidate = filter_qk(qk_candidate, query_lms)
        save_to_pickle(filtered_qk_candidate, "perspective_qk_candidate_filtered_{}".format(split))


if __name__ == "__main__":
    main()
