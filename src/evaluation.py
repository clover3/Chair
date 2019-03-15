from misc_lib import *
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import roc_curve
from sklearn import metrics

def top_k_idx(arr, k):
    return np.flip(np.argsort(arr))[:k]

def bottom_k_idx(arr, k):
    return np.argsort(arr)[:k]


def p_at_k_list_inner(explains, golds, k):
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
    return score_list_p , score_list_h

def p_at_k_list(explains, golds, k):
    score_list_p, score_list_h = p_at_k_list_inner(explains, golds, k)
    return average(score_list_p + score_list_h)


def p_at_k_list_ind(explains, golds, k):
    score_list_p, score_list_h = p_at_k_list_inner(explains, golds, k)
    return average(score_list_p), average(score_list_h)

def p_at_s_list(explains, golds):
    def p_at_stop(rank_list, gold_set):
        tp = 0
        g_size = len(gold_set)
        if g_size == 0:
            return 1
        for score, e in rank_list[:g_size]:
            if e in gold_set:
                tp += 1
        return tp / g_size

    score_list_h = []
    score_list_p = []
    for pred, gold in zip(explains, golds):
        pred_p, pred_h = pred
        gold_p, gold_h = gold
        if gold_p:
            s1 = p_at_stop(pred_p, gold_p)
            score_list_p.append(s1)
        if gold_h:
            s2 = p_at_stop(pred_h, gold_h)
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
                prec = (tp / n_pred_pos)
                assert prec <= 1
                sum += prec
        assert sum <= len(gold)
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


def MAP_rank(preds, golds):
    def AP(pred, gold):
        n_pred_pos = 0
        tp = 0
        sum_prec = 0
        for idx in pred:
            n_pred_pos += 1
            if gold[idx] == 1:
                tp += 1
                sum_prec += (tp / n_pred_pos)
        return sum_prec / sum(gold)

    scores = []
    for pred, gold, in zip(preds, golds):
        if sum(gold) > 0:
            scores.append(AP(pred, gold))

    return average(scores)


def MAP_inner(explains, golds):
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
    return score_list_p , score_list_h



def MAP_ind(explains, golds):
    score_list_p, score_list_h = MAP_inner(explains, golds)
    return average(score_list_p), average(score_list_h)



def PR_AUC_ind(explains, golds):
    def per_inst_AUC(pred, gold):
        n_pred_pos = 0
        tp = 0

        y_pred_list = []
        y_gold_list = []
        tie_break = 1

        all_pred_set = set([e for _, e in pred])
        for ge in gold:
            if ge not in all_pred_set:
                print("WARNING {} is not in {}".format(ge, all_pred_set))

        for score, e in pred:
            y_pred_list.append(score + tie_break)
            tie_break -= 0.01
            n_pred_pos += 1
            if e in gold:
                tp += 1
                y_gold_list.append(1)
            else:
                y_gold_list.append(0)

        prec_list, recall_list, _ = precision_recall_curve(y_gold_list, y_pred_list)
        r = auc(recall_list,prec_list)
        return r

    score_list_h = []
    score_list_p = []
    for pred, gold in zip(explains, golds):
        pred_p, pred_h = pred
        gold_p, gold_h = gold
        if gold_p:
            s1 = per_inst_AUC(pred_p, gold_p)
            score_list_p.append(s1)
        if gold_h:
            s2 = per_inst_AUC(pred_h, gold_h)
            score_list_h.append(s2)
    return average(score_list_p), average(score_list_h)


def PR_AUC(explains, golds):
    p_auc, h_auc = PR_AUC_ind(explains, golds)
    return (p_auc + h_auc) / 2


def compute_auc(y_scores, y_true):
    assert len(y_scores) == len(y_true)
    pairs = list(zip(y_scores, y_true))
    pairs.sort(key=lambda x:x[0], reverse=True)
    y_scores, y_true = zip(*pairs)

    p, r, thresholds = roc_curve(y_true, y_scores)
    return metrics.auc(p, r)


def compute_pr_auc(y_scores, y_true):
    assert len(y_scores) == len(y_true)
    pairs = list(zip(y_scores, y_true))
    pairs.sort(key=lambda x:x[0], reverse=True)
    y_scores, y_true = zip(*pairs)
    prec_list, recall_list, _ = precision_recall_curve(y_true, y_scores)
    return auc(recall_list, prec_list)
