from typing import List, Iterable, Callable, Dict, Tuple, Set

import json, os
import random
from cpath import output_path, data_path
from dataset_specific.msmarco.passage_common import enum_two_column_tsv
from misc_lib import path_join, get_dir_files
from trec.trec_parse import write_trec_ranked_list_entry
from trec.types import TrecRankedListEntry


def main():
    run_name = "manual"
    save_path = path_join(output_path, "transparency", "msmarco", "my_ranking.txt")

    rerank_dir = path_join(output_path, "transparency", "msmarco", "rerank_by_me")
    ranked_list = []
    for file_path in get_dir_files(rerank_dir):
        qid = os.path.basename(file_path)
        lines = open(file_path, "r").readlines()

        doc_id_list = []
        for line in lines[1:]:
            try:
                doc_id = int(line)
                doc_id_list.append(doc_id)
            except ValueError:
                pass

        if len(doc_id_list) != 10:
            raise Exception

        for rank, doc_id in enumerate(doc_id_list):
            score = 10 - rank
            e = TrecRankedListEntry(qid, str(doc_id), rank+1, score, run_name)
            ranked_list.append(e)

    write_trec_ranked_list_entry(ranked_list, save_path)


if __name__ == "__main__":
    main()