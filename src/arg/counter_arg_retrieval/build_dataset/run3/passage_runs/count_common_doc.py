import os
from collections import Counter

from cpath import output_path
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry


def main():
    def get_ranked_list(run_name):
        save_path = os.path.join(output_path, "ca_building",
                                 "passage_ranked_list", "{}.txt".format(run_name))
        return load_ranked_list_grouped(save_path)

    run_name_list = ["PQ_2", "PQ_3", "PQ_4", "PQ_5"]
    run_name_list = ["PQ_6", "PQ_7", "PQ_8", "PQ_9"]
    rlg_list = list(map(get_ranked_list, run_name_list))

    n_queries = 7
    queries = list(rlg_list[0].keys())
    assert len(queries) == n_queries
    for top_k in [20, 50, 100]:
        n_docs_per_query = []
        unique_counter = Counter()
        for qid in queries:
            doc_ids_per_query = set()
            counter = Counter()
            for rlg in rlg_list:
                sliced = rlg[qid][:top_k]
                doc_ids = list(map(TrecRankedListEntry.get_doc_id, sliced))
                doc_ids_per_query.update(doc_ids)
                counter.update(doc_ids)

            for idx, rlg in enumerate(rlg_list):
                sliced = rlg[qid][:top_k]
                doc_ids = list(map(TrecRankedListEntry.get_doc_id, sliced))
                n_unique = 0
                for doc in doc_ids:
                    if counter[doc] == 1:
                        n_unique += 1
                unique_counter[idx] += n_unique
            n_docs_per_query.append(len(doc_ids_per_query))

        n_num_doc_max = n_queries * top_k * len(rlg_list)

        run_overlap = sum(n_docs_per_query)
        print(f"num_queries={n_queries} top-k={top_k} num_runs={len(rlg_list)}")
        print("Query overlap {0} ({1:.2f})".
              format(run_overlap, run_overlap / n_num_doc_max))
        print(n_docs_per_query)
        print("(Max {})".format(n_num_doc_max))
        for idx, run_name in enumerate(run_name_list):
            print(idx, run_name, unique_counter[idx])


if __name__ == "__main__":
    main()

