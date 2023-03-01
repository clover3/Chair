

import json
import random
from cpath import output_path, data_path
from list_lib import index_by_fn
from misc_lib import path_join, get_first
from trec.trec_parse import write_trec_relevance_judgement
from trec.types import TrecRelevanceJudgementEntry


def main():
    qrel_path = path_join(data_path, "splade", "msmarco", "TREC_DL_2019", "qrel.json")

    candidate_ids_path = path_join(output_path, "transparency", "msmarco", "rerank_candiate_ids.json")
    candidate_ids = json.load(open(candidate_ids_path, "r"))
    candidate_ids_d = index_by_fn(get_first, candidate_ids)

    qrel = json.load(open(qrel_path, "r"))
    e_list = []
    for qid in qrel:
        if qid not in candidate_ids_d:
            continue
        qid_, pos_doc_ids, neg_doc_ids = candidate_ids_d[qid]
        for doc_id, value in qrel[qid].items():
            if doc_id in pos_doc_ids or doc_id in neg_doc_ids:
                e = TrecRelevanceJudgementEntry(qid, doc_id, value)
                if doc_id in pos_doc_ids:
                    assert value > 0
                elif doc_id in neg_doc_ids:
                    assert value == 0
                e_list.append(e)

    qrel_save_path = path_join(output_path, "transparency", "msmarco", "TREC_DL_2019_mini_qrel.txt")
    write_trec_relevance_judgement(e_list, qrel_save_path)



if __name__ == "__main__":
    main()
