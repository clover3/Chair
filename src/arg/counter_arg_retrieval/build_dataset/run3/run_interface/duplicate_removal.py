# Remove duplicate documents
import os
from typing import List, Tuple

# Input: Ranked List, SWTT
# Output: Ranked List
from arg.counter_arg_retrieval.build_dataset.data_prep.duplicate_removal import remove_duplicates
from bert_api.swtt.segmentwise_tokenized_text import SegmentwiseTokenizedText
from cache import load_from_pickle
from cpath import output_path
from list_lib import flatten
from trec.trec_parse import load_ranked_list_grouped, write_trec_ranked_list_entry


def main():
    rlg_path = os.path.join(output_path, "ca_building", "run3", "q_res_2.txt")
    rlg = load_ranked_list_grouped(rlg_path)
    docs: List[Tuple[str, SegmentwiseTokenizedText]] = load_from_pickle("ca_run3_swtt")
    docs, duplicate_doc_ids, new_rlg = remove_duplicates(rlg, docs)
    rl_itr = flatten(new_rlg.values())
    rlg_path = os.path.join(output_path, "ca_building", "run3", "q_res_2.filtered.txt")
    write_trec_ranked_list_entry(rl_itr, rlg_path)


if __name__ == "__main__":
    main()