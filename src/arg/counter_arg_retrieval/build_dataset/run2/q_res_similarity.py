import os
from typing import List, Dict, Set

from arg.counter_arg_retrieval.build_dataset.run2.load_data import load_my_run2_topics, CAQuery
from cpath import output_path
from list_lib import lmap
from tab_print import print_table
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry


def main():
    def load_ranked_list_from_dir(save_name):
        save_path = os.path.join(output_path, "ca_building", "run2", save_name)
        return load_ranked_list_grouped(save_path)

    rlg_ca: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_from_dir("rerank_pers.txt")
    rlg_pers: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_from_dir("rerank_ca.txt")

    topic_d: Dict[str, CAQuery] = {topic.qid: topic for topic in load_my_run2_topics()}

    rows = []
    for qid in rlg_ca.keys():
        def get_top_k_docs(rlg, k) -> Set:
            l = rlg[qid]
            top50_from = lmap(TrecRankedListEntry.get_doc_id, l[:k])
            return set(top50_from)

        print(f"Qid={qid}")
        print("Perspective: ", topic_d[qid].perspective)
        print("Counter-argument: ", topic_d[qid].ca_query)

        row = [qid, topic_d[qid].perspective, topic_d[qid].ca_query]
        for k in [5, 10, 50, 100]:
            ca_choice = get_top_k_docs(rlg_ca, k)
            pers_choice = get_top_k_docs(rlg_pers, k)
            common = ca_choice.intersection(pers_choice)
            n_common = len(common)/k
            row.append(n_common)

        rows.append(row)
    print_table(rows)



if __name__ == "__main__":
    main()