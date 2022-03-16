from typing import List, Callable


def is_tp(pred: bool, label: bool) -> bool:
    return pred and label


def is_fp(pred: bool, label: bool):
    return pred and not label


def is_tn(pred: bool, label: bool):
    return not pred and not label


def is_fn(pred: bool, label: bool):
    return not pred and label


def get_true_positive_list(binary_scores: List[int], labels: List[int]):
    return _list_common_compare(binary_scores, labels, is_tp)


def get_true_negative_list(binary_scores: List[int], labels: List[int]):
    return _list_common_compare(binary_scores, labels, is_tn)


def get_false_positive_list(binary_scores: List[int], labels: List[int]):
    return _list_common_compare(binary_scores, labels, is_fp)


def get_false_negative_list(binary_scores: List[int], labels: List[int]):
    return _list_common_compare(binary_scores, labels, is_fn)


def _list_common_compare(binary_scores: List[int], labels: List[int],
                         compare_fn: Callable[[bool, bool], bool]) -> List[int]:
    return [int(compare_fn(bool(p), bool(l))) for p, l in zip(binary_scores, labels)]