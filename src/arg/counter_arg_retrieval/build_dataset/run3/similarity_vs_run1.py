import os
from typing import List, Dict, Set

from cpath import output_path
from list_lib import lmap
from tab_print import print_table
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry


def main():
    qid_pair_list = [("218", "47"), ("78", "504")]

    save_path = os.path.join(output_path, "ca_building", "run3", "q_res_2.txt")
    rows = []
    rlg_run3: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(save_path)
    for qid_run1, qid_run3 in qid_pair_list:
        save_path = os.path.join(output_path, "ca_building", "q_res", "q_res_qid{}.txt".format(qid_run1))
        rlg_run1: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(save_path)
        rl1 = rlg_run1[qid_run1]
        rl3 = rlg_run3[qid_run3]

        def get_top_k_docs(rl, k) -> Set:
            top50_from = lmap(TrecRankedListEntry.get_doc_id, rl[:k])
            return set(top50_from)
        print(f"Qid={qid_run3}")
        print("Ranked list length: {} , {}".format(len(rl1), len(rl3)))

        row = []
        for k in [5, 10, 50, 100, 1000]:
            run1_choice = get_top_k_docs(rl1, k)
            run3_choice = get_top_k_docs(rl3, k)
            common = run1_choice.intersection(run3_choice)
            n_common = len(common)/k
            row.append(n_common)

        rows.append(row)
    print_table(rows)


if __name__ == "__main__":
    main()