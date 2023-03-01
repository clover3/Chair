from typing import List, Iterable, Callable, Dict, Tuple, Set

import json
import random
from cpath import output_path, data_path
from dataset_specific.msmarco.passage_common import enum_two_column_tsv
from misc_lib import path_join


def main():
    candidates_path = path_join(output_path, "transparency", "msmarco", "rerank_candiate_ids.json")
    candidate_ids: List[Tuple[str, List[str], List[str]]] = json.load(open(candidates_path, "r"))
    collection_path = path_join(data_path, "splade", "msmarco", "full_collection", "raw.tsv")
    all_doc_ids = set()

    for qid, pos_ids, neg_ids in candidate_ids:
        all_doc_ids.update(pos_ids)
        all_doc_ids.update(neg_ids)

    passage_d = {}
    for passage_id, text in enum_two_column_tsv(collection_path):
        if passage_id in all_doc_ids:
            passage_d[passage_id] = text
            print(passage_id)


    print("{} documents found".format(len(passage_d)))
    passage_data = path_join(output_path, "transparency", "msmarco", "rerank_passages.json")
    json.dump(passage_d, open(passage_data, "w"))


if __name__ == "__main__":
    main()