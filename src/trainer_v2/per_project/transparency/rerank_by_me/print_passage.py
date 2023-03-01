from typing import List, Iterable, Callable, Dict, Tuple, Set

import json
import random
from cpath import output_path, data_path
from dataset_specific.msmarco.passage_common import enum_two_column_tsv
from misc_lib import path_join


def main():
    candidates_path = path_join(output_path, "transparency", "msmarco", "rerank_candiate_ids.json")
    passage_path = path_join(output_path, "transparency", "msmarco", "rerank_passages.json")
    rerank_dir = path_join(output_path, "transparency", "msmarco", "rerank_jobs")
    query_path = path_join(data_path, "splade", "msmarco", "TREC_DL_2019", "queries_2019", "raw.tsv")

    query_d = {}
    for qid, text in enum_two_column_tsv(query_path):
        query_d[qid] = text

    passage_data = json.load(open(passage_path, "r"))
    candidate_ids: List[Tuple[str, List[str], List[str]]] = json.load(open(candidates_path, "r"))

    for qid, pos_ids, neg_ids in candidate_ids:
        save_path = path_join(rerank_dir, qid)
        all_doc_ids = pos_ids + neg_ids
        all_doc_ids.sort()
        query = query_d[qid]
        lines = []
        lines.append(query)
        lines.append("---")
        for doc_id in all_doc_ids:
            passage = passage_data[doc_id]
            lines.append(doc_id)
            lines.append(passage)
            lines.append("---")

        lines = [l + "\n" for l in lines]
        f_out = open(save_path, "w")
        f_out.writelines(lines)



if __name__ == "__main__":
    main()