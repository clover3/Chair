import sys
from collections import defaultdict
from typing import List, Dict, Tuple

from arg.perspectives.eval_caches import get_eval_candidate_as_pids
from arg.perspectives.load import get_claim_perspective_id_dict2
from evals.trec import load_ranked_list


def main():
    trec_path = sys.argv[1]
    ranked_list = load_ranked_list(trec_path)
    candidate_d_raw: Dict[Tuple[int, List[int]]] = dict(get_eval_candidate_as_pids("dev"))
    label_d: Dict[int, List[int]] = get_claim_perspective_id_dict2()

    ex_candiate_entry = defaultdict(list)
    for entry in ranked_list:
        cid = int(entry.query_id)
        pid = int(entry.doc_id)
        label = pid in label_d[cid]

        # show entry which are true and not in original candidate
        if label and pid not in candidate_d_raw[cid]:
            ex_candiate_entry[cid].append(entry.rank)

    for cid, ranks in ex_candiate_entry.items():
        print(cid, ranks)

if __name__ == "__main__":
    main()
