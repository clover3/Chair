import sys
from typing import List, Dict

from trec.trec_parse import load_ranked_list_grouped, TrecRankedListEntry
from scipy.stats import stats

from galagos.parse import load_qrels
from misc_lib import average
from runnable.trec.trec_eval_like import get_metric_fn


def get_score_per_query(qrels, metric_fn, ranked_list):
    score_per_query_dict = {}
    not_found = 0
    for query_id in ranked_list:
        q_ranked_list = ranked_list[query_id]

        try:
            gold_list = qrels[query_id]
            true_gold = list([doc_id for doc_id, score in gold_list if score > 0])
            score_per_query = metric_fn(q_ranked_list, true_gold)
            score_per_query_dict[query_id] = score_per_query
        except KeyError as e:
            not_found += 1
    if not_found:
        print("{} of {} queires not found".format(not_found, len(ranked_list)))

    return score_per_query_dict


def main():
    judgment_path = sys.argv[1]
    metric = sys.argv[2]
    ranked_list_path1 = sys.argv[3]
    ranked_list_path2 = sys.argv[4]
    # print
    qrels = load_qrels(judgment_path)

    ranked_list_1: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(ranked_list_path1)
    ranked_list_2: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(ranked_list_path2)

    metric_fn = get_metric_fn(metric)

    score_d1 = get_score_per_query(qrels, metric_fn, ranked_list_1)
    score_d2 = get_score_per_query(qrels, metric_fn, ranked_list_2)

    pairs = []
    for key in score_d1:
        try:
            e = (score_d1[key], score_d2[key])
            pairs.append(e)
        except KeyError as e:
            pass

    if len(pairs) < len(score_d1) or len(pairs) < len(score_d2):
        print("{} matched from {} and {} scores".format(len(pairs), len(score_d1), len(score_d2)))

    l1, l2 = zip(*pairs)
    d, p_value = stats.ttest_rel(l1, l2)
    print("baseline:", average(l1))
    print("treatment:", average(l2))
    print(d, p_value)


if __name__ == "__main__":
    main()