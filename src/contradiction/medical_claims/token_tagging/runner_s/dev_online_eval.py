import random
from typing import List, Dict, Tuple

from contradiction.medical_claims.label_structure import AlamriLabel, PairedIndicesLabel
from contradiction.medical_claims.token_tagging.eval_analyze.online_eval import load_sbl_labels
from contradiction.medical_claims.token_tagging.nli_interface import get_nli_cache_client
from contradiction.medical_claims.token_tagging.problem_loader import AlamriProblem, \
    load_alamri_split
from contradiction.medical_claims.token_tagging.solvers.intersection_solver import IntersectionSolver
from evals.prec_recall import get_true_positive_list, get_false_positive_list, is_fn, is_tn, is_fp, is_tp, \
    get_true_negative_list, get_false_negative_list

old_print = print

def print(s: str):
    old_print("dev_online_eval.py: " + s)



def display(p: AlamriProblem,
            score_tuple: Tuple[List[float], List[float]],
            label_tuples: Tuple[List[int], List[int]]):
    # TODO show what is prediction what is wrong
    def display_inner(scores: List[float], labels: List[int], text: str):
        binary_pred: List[bool] = [s >= 0.5 for s in scores]
        tokens: List[str] = text.split()
        maybe_len = len(tokens)
        if maybe_len > len(labels):
            pad_len = maybe_len - len(labels)
            labels = labels + [False for _ in range(pad_len)]
        elif maybe_len != len(labels):
            print("WARNING label has length {} while {} is expected".format(len(labels), maybe_len))
        elif maybe_len != len(scores):
            print("WARNING scores has length {} while {} is expected".format(len(scores), maybe_len))

        n_tp = sum(get_true_positive_list(binary_pred, labels))
        n_fp = sum(get_false_positive_list(binary_pred, labels))
        n_tn = sum(get_true_negative_list(binary_pred, labels))
        n_fn = sum(get_false_negative_list(binary_pred, labels))

        print(str(scores))
        print("tp/fp/tn/fn = {}/{}/{}/{}".format(n_tp, n_fp, n_tn, n_fn))
        def get_s(token: str, pred: bool, label: bool):
            if is_tn(pred, label):
                return "{}".format(token)
            elif is_tp(pred, label):
                return "[{}]".format(token)
            elif is_fp(pred, label):
                return "[{}]fp".format(token)
            elif is_fn(pred, label):
                return "[{}]fn".format(token)
            else:
                raise ValueError

        print(" ".join([get_s(tokens[i], binary_pred[i], bool(labels[i])) for i in range(maybe_len)]))

    display_inner(score_tuple[0], label_tuples[0], p.text1)
    display_inner(score_tuple[1], label_tuples[1], p.text2)
    print("")


def main():
    tag_type = "mismatch"
    split = "dev"
    random.seed(0)
    problems: List[AlamriProblem] = load_alamri_split(split)
    labels: List[AlamriLabel] = load_sbl_labels(split)
    labels_d: Dict[Tuple[int, int], PairedIndicesLabel] = {(l.group_no, l.inner_idx): l.label for l in labels}
    solver = get_interaction_solver()

    for p in problems[8:]:
        try:
            print("sent1: " + p.text1)
            print("sent2: " + p.text2)
            score_tuple: Tuple[List[float], List[float]] = solver.solve_from_text(p.text1, p.text2)
            label: PairedIndicesLabel = labels_d[p.group_no, p.inner_idx]
            label_tuples = label.get_label_tuple(tag_type)
            display(p, score_tuple, label_tuples)
        except KeyError:
            pass
        break


def get_interaction_solver():
    cache_client = get_nli_cache_client("localhost")
    solver = IntersectionSolver(cache_client.predict, True)
    return solver


if __name__ == "__main__":
    main()