from misc_lib import *


def p_at_k_list(explains, golds, k):
    def p_at_k(rank_list, gold_set, k):
        tp = 0
        for score, e in rank_list[:k]:
            if e in gold_set:
                tp += 1
        return tp / k

    score_list_h = []
    score_list_p = []
    for pred, gold in zip(explains, golds):
        pred_p, pred_h = pred
        gold_p, gold_h = gold
        if gold_p:
            s1 = p_at_k(pred_p, gold_p, k)
            score_list_p.append(s1)
        if gold_h:
            s2 = p_at_k(pred_h, gold_h, k)
            score_list_h.append(s2)
    return average(score_list_p + score_list_h)


def MAP(explains, golds):
    def AP(pred, gold):
        n_pred_pos = 0
        tp = 0
        sum = 0
        for score, e in pred:
            n_pred_pos += 1
            if e in gold:
                tp += 1
                sum += (tp / n_pred_pos)
        return sum / len(gold)

    score_list_h = []
    score_list_p = []
    for pred, gold in zip(explains, golds):
        pred_p, pred_h = pred
        gold_p, gold_h = gold
        if gold_p:
            s1 = AP(pred_p, gold_p)
            score_list_p.append(s1)
        if gold_h:
            s2 = AP(pred_h, gold_h)
            score_list_h.append(s2)
    return average(score_list_p + score_list_h)