from typing import List, Tuple

from attribution.attrib_types import TokenScores
from cache import load_from_pickle
from contradiction.mnli_ex.runner.run_idf2_baseline import load_labels_as_tuples
from explain.eval_all import eval_nli_explain


def run_eval2():
    run_name = "pred_nli_ex_26_mismatch_1000"
    split = "test"
    target_label = "mismatch"
    predictions = load_from_pickle(run_name)
    predictions = predictions[100:700]
    print(predictions[0])
    gold_list: List[Tuple[List[int], List[int]]] = load_labels_as_tuples(split, target_label)
    scores = eval_nli_explain(predictions, gold_list, False, True)
    print(scores)


def p1_debug(predictions: List[Tuple[TokenScores, TokenScores]],golds: List[Tuple[List[int], List[int]]]):
    k = 1
    def p_at_k(rank_list, gold_set, k):
        tp = 0
        for score, e in rank_list[:k]:
            if e in gold_set:
                tp += 1
        return tp / k

    score_list_h = []
    score_list_p = []
    for pred, gold in zip(predictions, golds):
        pred_p, pred_h = pred
        gold_p, gold_h = gold
        if gold_p:
            s1 = p_at_k(pred_p, gold_p, k)
            score_list_p.append(s1)
        if gold_h:
            _, top_pred = pred_h[0]
            s2 = p_at_k(pred_h, gold_h, k)
            print(top_pred, gold_h, s2)
            score_list_h.append(s2)

    print(score_list_h)