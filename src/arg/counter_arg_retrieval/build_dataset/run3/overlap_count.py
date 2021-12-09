import os

from cpath import output_path
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry


def main():

    rlg_path = os.path.join(output_path, "ca_building", "run3", "q_res_2.txt")
    rlg1 = load_ranked_list_grouped(rlg_path)
    rlg_path = os.path.join(output_path, "ca_building", "run3", "pc_res.txt")
    rlg2 = load_ranked_list_grouped(rlg_path)

    for qid in rlg2:
        cid, pid = qid.split("_")
        rl1 = rlg1[cid]
        rl2 = rlg2[qid]
        doc_ids1 = set(map(TrecRankedListEntry.get_doc_id, rl1))
        doc_ids2 = set(map(TrecRankedListEntry.get_doc_id, rl2))
        n_common = len(doc_ids1.intersection(doc_ids2))
        print(qid, n_common, len(rl1), len(rl2))


if __name__ == "__main__":
    main()
