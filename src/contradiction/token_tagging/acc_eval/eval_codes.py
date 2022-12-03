from collections import defaultdict
from typing import List, Dict

from contradiction.token_tagging.acc_eval.defs import SentTokenLabel, SentTokenBPrediction
from evals.basic_func import get_acc_prec_recall, i2b
from list_lib import index_by_fn
from misc_lib import average


def calculate_acc_inner(label_list: List[SentTokenLabel], prediction_list: List[SentTokenBPrediction]):
    label_d = index_by_fn(lambda x: x.qid, label_list)
    acc_per_sent: List[float] = []
    number_of_sents = 0
    for p in prediction_list:
        try:
            labels = label_d[p.qid].labels
            predictions = p.predictions

            if len(labels) != len(predictions):
                print("WARNING number of tokens differ: {} != {}".format(len(labels), len(predictions)))

            correctness: List[int] = []
            for i, l in enumerate(labels):
                try:
                    if l == predictions[i]:
                        correct = 1
                    else:
                        correct = 0
                except IndexError:
                    correct = 0
                correctness.append(correct)

            acc_per_sent.append(average(correctness))
            number_of_sents += 1
        except KeyError:
            pass

    return average(acc_per_sent), number_of_sents


def calc_prec_rec_acc(label_list: List[SentTokenLabel], prediction_list: List[SentTokenBPrediction]):
    label_d = index_by_fn(lambda x: x.qid, label_list)
    scores_list = defaultdict(list)
    number_of_sents = 0
    for p in prediction_list:
        try:
            labels = label_d[p.qid].labels
            predictions = p.predictions

            if len(labels) != len(predictions):
                print("WARNING number of tokens differ: {} != {}".format(len(labels), len(predictions)))

            per_problem: Dict = get_acc_prec_recall(i2b(predictions), i2b(labels))
            for key in per_problem:
                scores_list[key].append(per_problem[key])
            number_of_sents += 1
        except KeyError:
            pass
    metrics = {}
    for key in scores_list:
        if key in ["accuracy", "precision", "recall", "f1"]:
            metrics[key] = average(scores_list[key])
        elif key in ["tp", "tn", "fp", "fn"]:
            metrics[key] = sum(scores_list[key])
        else:
            raise KeyError

    metrics['number_of_sents'] = number_of_sents
    return metrics


def calculate_acc(label_list: List[SentTokenLabel], prediction_list: List[SentTokenBPrediction]) -> float:
    acc, number_of_sents = calculate_acc_inner(label_list, prediction_list)
    print("{} sentences evaluated".format(number_of_sents))
    return acc