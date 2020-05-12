import os
import pickle
import sys
from typing import List, Dict

from arg.perspectives.declaration import ParagraphClaimPersFeature
from arg.perspectives.n_gram_feature_collector import remove_duplicate
from arg.pf_common.base import ScoreParagraph
from cache import save_to_pickle
from misc_lib import TimeEstimator


def merge(items: List[ParagraphClaimPersFeature]) -> Dict[str, List[ScoreParagraph]]:
    group_by_cid: Dict[str, List[ScoreParagraph]] = {}

    for item in items:
        cid = item.claim_pers.cid
        if cid not in group_by_cid:
            group_by_cid[cid] = []

        para_list = item.feature[:100]
        group_by_cid[cid].extend(para_list)

    group_by_cid_out: Dict[str, List[ScoreParagraph]] = {}
    for cid, para_list in group_by_cid.items():
        group_by_cid_out[cid] = list(remove_duplicate(para_list))
    return group_by_cid_out


def work(input_dir, st, ed, save_name):
    all_features = []
    ticker = TimeEstimator(ed-st)
    for i in range(st, ed):
        try:
            features: List[ParagraphClaimPersFeature] = pickle.load(open(os.path.join(input_dir, str(i)), "rb"))
            all_features.extend(features)
        except FileNotFoundError:
            pass
        ticker.tick()

    print("merging...")
    out: Dict[str, List[ScoreParagraph]] = merge(all_features)
    save_to_pickle(out, save_name)


if __name__ == "__main__":
    work(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
