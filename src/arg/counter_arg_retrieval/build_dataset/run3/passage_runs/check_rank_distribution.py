import os

from cpath import output_path
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry


def main():
    rlg_path = os.path.join(output_path, "ca_building", "run3", "q_res_2.filtered.dummy.txt")
    rlg_source = load_ranked_list_grouped(rlg_path)

    top_k = 100

    def get_ranked_list(run_name):
        save_path = os.path.join(output_path, "ca_building", "run3",
                                 "passage_ranked_list", "{}.txt".format(run_name))
        return load_ranked_list_grouped(save_path)

    run_name_list = ["PQ_2", "PQ_3", "PQ_4", "PQ_5"]

    for run_name in run_name_list:
        rlg = get_ranked_list(run_name)
        print(run_name)
        for qid, entries in rlg.items():
            source_doc_list = list(map(TrecRankedListEntry.get_doc_id, rlg_source[qid]))
            sliced = entries[:top_k]
            passage_doc_ids = list(map(TrecRankedListEntry.get_doc_id, sliced))
            doc_ids = [p_doc_id.split("_")[0] for p_doc_id in passage_doc_ids]
            rank_in_original = [source_doc_list.index(doc_id) for doc_id in doc_ids]
            print(qid)
            for step in [100, 1000, 10000]:
                num_included = len([r for r in rank_in_original if r < step])
                print(step, num_included)



if __name__ == "__main__":
    main()