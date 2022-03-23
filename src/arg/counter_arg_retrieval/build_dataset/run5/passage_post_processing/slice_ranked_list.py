import os
import sys
from typing import List, Dict

from arg.counter_arg_retrieval.build_dataset.run5.query_enum_util import get_files_for_each_query
from cpath import output_path
from trec.trec_parse import load_ranked_list_grouped, write_trec_ranked_list_entry
from trec.types import TrecRankedListEntry


def slice_for_run(run_name, k) -> List[TrecRankedListEntry]:
    old_rlg_dir = os.path.join(output_path, "ca_building", "run5", "passage_ranked_list_per_query", run_name)
    file_list = get_files_for_each_query(old_rlg_dir)
    new_ranked_list = []
    for file_path in file_list:
        rl: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(file_path)
        for qid, entries in rl.items():
            new_ranked_list.extend(entries[:k])
    return new_ranked_list


def main():
    run_name = sys.argv[1]
    new_ranked_list: List[TrecRankedListEntry] = slice_for_run(run_name, 100)
    save_path = os.path.join(output_path, "ca_building",
                             "passage_ranked_list_100", '{}.txt'.format(run_name))

    write_trec_ranked_list_entry(new_ranked_list, save_path)


if __name__ == "__main__":
    main()
