import sys
from typing import List, Dict

from list_lib import left
from trec.qrel_parse import load_qrels_flat_per_query
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry


def main():
    judgment_path = sys.argv[1]
    ranked_list_path = sys.argv[2]

    qrels = load_qrels_flat_per_query(judgment_path)
    ranked_list: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(ranked_list_path)
    not_found = 0
    n_entry = 0
    for query_id in ranked_list:
        q_ranked_list = ranked_list[query_id]
        gold_list = qrels[query_id]
        judged_doc_id = left(gold_list)

        for e in q_ranked_list:
            n_entry += 1
            if e.doc_id not in judged_doc_id:
                not_found += 1

    print("{} of {} query-doc not found".format(not_found, n_entry))





if __name__ == "__main__":
    main()