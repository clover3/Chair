import sys
from typing import List, Dict

from trec.parse import load_qrels_flat

from evals.metrics import get_metric_fn
from misc_lib import average
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry


## print


def main():
    judgment_path = sys.argv[1]
    ranked_list_path = sys.argv[2]
    metric = sys.argv[3]

    qrels = load_qrels_flat(judgment_path)
    ranked_list: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(ranked_list_path)

    metric_fn = get_metric_fn(metric)

    score_per_query_list = []
    not_found = 0
    for query_id in ranked_list:
        q_ranked_list = ranked_list[query_id]

        try:
            gold_list = qrels[query_id]
            true_gold = list([doc_id for doc_id, score in gold_list if score > 0])
            score_per_query = metric_fn(q_ranked_list, true_gold)
            score_per_query_list.append(score_per_query)
        except KeyError as e:
            not_found += 1

    if not_found:
        print("{} of {} queires not found".format(not_found, len(ranked_list)))

    score = average(score_per_query_list)
    print("{}\t{}".format(metric, score))


if __name__ == "__main__":
    main()