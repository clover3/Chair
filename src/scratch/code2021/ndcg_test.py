import sys
from typing import List, Dict

from trec.parse import load_qrels_flat

from misc_lib import average
from scratch.ndcg_lib import dcg
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry


def main():
    judgment_path = sys.argv[1]
    ranked_list_path1 = sys.argv[2]
    # print
    qrels = load_qrels_flat(judgment_path)

    ranked_list: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(ranked_list_path1)
    print(37)

    all_scores_list = []

    all_label_list = []
    per_score_list = []
    k = 20
    for query_id in ranked_list:
        q_ranked_list = ranked_list[query_id]
        try:
            gold_list = qrels[query_id]
            true_gold = list([doc_id for doc_id, score in gold_list if score > 0])

            label_list = []
            seen_docs = []
            for e in q_ranked_list:
                label = 1 if e.doc_id in true_gold else 0
                seen_docs.append(e.doc_id)
                label_list.append(label)

            d1 = dcg(label_list[:k])
            n_true = min(len(true_gold), k)
            idcg = dcg([1] * n_true)
            per_score = d1 / idcg if true_gold else 1
            for doc_id in true_gold:
                if doc_id not in seen_docs:
                    label_list.append(1)
        except KeyError as e:
            per_score = 1
            print("Query not found:", query_id)
        per_score_list.append(per_score)

    print(average(per_score_list))

if __name__ == "__main__":
    main()