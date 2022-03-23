import os
import sys
from typing import List, Dict

from arg.counter_arg_retrieval.build_dataset.path_helper import get_sliced_ranked_list_path
from cpath import output_path
from trec.trec_parse import load_ranked_list_grouped, write_trec_ranked_list_entry
from trec.types import TrecRankedListEntry


def slice_for_run(run_name, k) -> List[TrecRankedListEntry]:
    file_path = os.path.join(output_path, "ca_building", "passage_ranked_list_100_unique", "{}.txt".format(run_name))
    new_ranked_list = []
    rl: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(file_path)
    for qid, entries in rl.items():
        if len(entries) < k:
            print("WARNING query {} has only {} items".format(qid, len(entries)))
        new_ranked_list.extend(entries[:k])
    return new_ranked_list


def main():
    run_name = sys.argv[1]
    new_ranked_list: List[TrecRankedListEntry] = slice_for_run(run_name, 20)
    save_path = get_sliced_ranked_list_path(run_name)
    write_trec_ranked_list_entry(new_ranked_list, save_path)


if __name__ == "__main__":
    main()
