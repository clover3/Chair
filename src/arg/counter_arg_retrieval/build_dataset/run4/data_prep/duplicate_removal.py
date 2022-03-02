# Remove duplicate documents
import os
import pickle
from typing import List, Tuple

from arg.counter_arg_retrieval.build_dataset.data_prep.duplicate_removal import remove_duplicates
from bert_api.swtt.segmentwise_tokenized_text import SegmentwiseTokenizedText
from cpath import output_path
from list_lib import flatten
from trec.trec_parse import load_ranked_list_grouped, write_trec_ranked_list_entry


def load_docs():
    root_dir = os.path.join(output_path, "ca_building", "run3")
    max_job = 70
    run_name = "pc_res_doc_jsonl_parsing"
    all_docs = []
    for i in range(max_job):
        pickle_path = os.path.join(root_dir, run_name, str(i))
        docs = pickle.load(open(pickle_path, "rb"))
        all_docs.extend(docs)

    print("Loaded {} docs".format(len(all_docs)))
    return all_docs


def main():
    rlg_path = os.path.join(output_path, "ca_building", "run4", "pc_res.txt")
    rlg = load_ranked_list_grouped(rlg_path)
    docs: List[Tuple[str, SegmentwiseTokenizedText]] = load_docs()
    docs, duplicate_doc_ids, new_rlg = remove_duplicates(rlg, docs)
    rl_itr = flatten(new_rlg.values())
    rlg_path = os.path.join(output_path, "ca_building", "run4", "pc_res.filtered.txt")
    write_trec_ranked_list_entry(rl_itr, rlg_path)


if __name__ == "__main__":
    main()