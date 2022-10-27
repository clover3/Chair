from collections import defaultdict
from typing import List, Dict

from cache import load_list_from_jsonl, save_list_to_jsonl_w_fn
from evals.basic_func import get_acc_prec_recall, i2b
from list_lib import index_by_fn
from misc_lib import average


class SentTokenLabel:
    def __init__(self, qid, labels):
        self.qid: str = qid
        self.labels: List[int] = labels

    def to_json(self):
        return {
            'qid': self.qid,
            'labels': self.labels,
        }

    @classmethod
    def from_json(cls, j):
        qid = j['qid']
        labels = j['labels']
        assert type(qid) == str
        for item in labels:
            assert type(item) == int
        return SentTokenLabel(qid, labels)


class SentTokenBPrediction:
    def __init__(self, qid, predictions):
        self.qid: str = qid
        self.predictions: List[int] = predictions

    def to_json(self):
        return {
            'qid': self.qid,
            'predictions': self.predictions,
        }

    @classmethod
    def from_json(cls, j):
        qid = j['qid']
        predictions = j['predictions']
        assert type(qid) == str
        for item in predictions:
            assert type(item) == int
        return SentTokenBPrediction(qid, predictions)


def save_sent_token_binary_predictions(predictions: List[SentTokenBPrediction], save_path):
    return save_list_to_jsonl_w_fn(predictions, save_path, SentTokenBPrediction.to_json)


def load_sent_token_binary_predictions(save_path):
    return load_list_from_jsonl(save_path, SentTokenBPrediction.from_json)


def load_sent_token_label(file_path) -> List[SentTokenLabel]:
    label_list: List[SentTokenLabel] = load_list_from_jsonl(file_path, SentTokenLabel.from_json)
    return label_list


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
