# Remove duplicate documents
import os
from typing import List, Tuple

# Input: Ranked List, SWTT
# Output: Ranked List
from bert_api.swtt.segmentwise_tokenized_text import SegmentwiseTokenizedText
from cache import load_from_pickle
from cpath import output_path
from list_lib import right, flatten
from trec.ranked_list_util import remove_duplicates_from_ranked_list
from trec.trec_parse import load_ranked_list_grouped, write_trec_ranked_list_entry


def remove_duplicates(ranked_list_grouped, docs: List[Tuple[str, SegmentwiseTokenizedText]]):
    n_docs = len(docs)
    duplicate_indices = SegmentwiseTokenizedText.get_duplicate(right(docs))
    print("duplicate_indices {} ".format(len(duplicate_indices)))
    duplicate_doc_ids = [docs[idx][0] for idx in duplicate_indices]
    docs = [e for idx, e in enumerate(docs) if idx not in duplicate_indices]
    print("{} docs after filtering (from {})".format(len(docs), n_docs))
    new_ranked_list_grouped = remove_duplicates_from_ranked_list(ranked_list_grouped, duplicate_doc_ids)
    return docs, duplicate_doc_ids, new_ranked_list_grouped


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