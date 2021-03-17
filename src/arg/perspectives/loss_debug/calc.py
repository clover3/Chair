import sys
from typing import List, Dict, Tuple

import math
from trec.parse import load_qrels_flat

from misc_lib import average
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry, QueryID, DocID


def main():
    first_list_path = sys.argv[1]
    second_list_path = sys.argv[2]
    print("Use {} if available, if not use {}".format(first_list_path, second_list_path))
    l1: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(first_list_path)
    l2: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(second_list_path)

    judgment_path = sys.argv[3]
    qrels: Dict[QueryID, List[Tuple[DocID, int]]] = load_qrels_flat(judgment_path)

    def eval_loss(prob, label):
        if label:
            loss = -math.log(prob)
        else:
            loss = -math.log(1-prob)
        return loss

    def get_loss(l):
        loss_all = []
        for query_id, ranked_list in l.items():
            gold_list = qrels[query_id]
            true_gold: List[str] = list([doc_id for doc_id, score in gold_list if score > 0])
            for e in ranked_list:
                label = e.doc_id in true_gold
                loss = eval_loss(e.score, label)
                loss_all.append(loss)

        return average(loss_all)

    def get_acc(l):
        correctness = []
        for query_id, ranked_list in l.items():
            gold_list = qrels[query_id]
            true_gold: List[str] = list([doc_id for doc_id, score in gold_list if score > 0])
            for e in ranked_list:
                label = e.doc_id in true_gold
                is_correct = (e.score > 0.5) == label
                # print(label, e.score, is_correct)
                correctness.append(int(is_correct))

        return average(correctness)

    def get_tf(l):
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for query_id, ranked_list in l.items():
            gold_list = qrels[query_id]
            true_gold: List[str] = list([doc_id for doc_id, score in gold_list if score > 0])
            for e in ranked_list:
                label = e.doc_id in true_gold
                pred_true = e.score > 0.5
                if pred_true:
                    if label:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if label:
                        tn += 1
                    else:
                        fn += 1

        return tp, fp, tn, fn

    print("loss1:", get_loss(l1))
    print("loss2:", get_loss(l2))

    print("acc1:", get_acc(l1))
    print("acc2:", get_acc(l2))

    print("1: tp fp tn fn ", get_tf(l1))
    print("2: tp fp tn fn :", get_tf(l2))


if __name__ == "__main__":
    main()
