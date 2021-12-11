import os

from cpath import output_path
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry


def main():
    def get_ranked_list(run_name):
        save_path = os.path.join(output_path, "ca_building", "run3",
                                 "passage_ranked_list", "{}.txt".format(run_name))
        return load_ranked_list_grouped(save_path)

    run_name_list = ["PQ_1", "PQ_3", "PQ_4"]
    run_name_list = ["PQ_2", "PQ_3", "PQ_4", "PQ_5"]
    rlg_list = list(map(get_ranked_list, run_name_list))

    n_queries = 7
    queries = list(rlg_list[0].keys())
    assert len(queries) == n_queries
    top_k = 20

    num_doc_per_qid = {}
    all_doc_ids = set()
    for qid in queries:
        for rlg in rlg_list:
            sliced = rlg[qid][:top_k]
            doc_ids = list(map(TrecRankedListEntry.get_doc_id, sliced))
            all_doc_ids.update(doc_ids)
        num_doc_per_qid[qid] = len(all_doc_ids)

    num_all_docs = len(all_doc_ids)
    n_num_doc_max = n_queries * top_k * len(rlg_list)

    print(f"num_queries={n_queries} top-k={top_k} num_runs={len(rlg_list)}")
    print("{} docs".format(num_all_docs))
    print("(Max {})".format(n_num_doc_max))


if __name__ == "__main__":
    main()