from dataset_specific.scientsbank.pte_solver_if import apply_solver, PTESolverIF, PTESolverAllTrue, PTESolverAllFalse
from dataset_specific.scientsbank.parse_fns import load_scientsbank_split, sci_ents_all_splits
from dataset_specific.scientsbank.pte_data_types import PTEPredictionPerQuestion, Question
from typing import List, Iterable, Callable, Dict, Tuple, Set

from dataset_specific.scientsbank.save_load_pred import save_pte_preds_to_file
from evals.basic_func import get_acc_prec_recall
from list_lib import lflatten, index_by_fn
from misc_lib import average
from tab_print import print_table


def convert_to_paired_list(
        questions: List[Question], preds: List[PTEPredictionPerQuestion])\
        -> List[List[List[Tuple[bool, bool]]]]:
    q_d = index_by_fn(lambda q: q.id, questions)
    lll = []
    for pred in preds:
        question = q_d[pred.id]
        ll = []
        assert question.id == pred.id
        for sa, sp in zip(question.student_answers, pred.per_student_answer_list):
            l = []
            assert sa.id == sp.id
            for fe, pe, in zip(sa.facet_entailments, sp.facet_pred):
                assert fe.facet_id == pe.facet_id
                l.append((fe.get_bool_label(), pe.pred))
            ll.append(l)
        lll.append(ll)
    return lll


def assert_all_equal(value_list):
    prev_value = None
    for v in value_list:
        if prev_value is not None:
            assert v == prev_value
        prev_value = v


def evaluate(questions: List[Question], preds: List[PTEPredictionPerQuestion]) -> dict:
    lll: List[List[List[Tuple[bool, bool]]]] = convert_to_paired_list(questions, preds)
    ll: List[List[Tuple[bool, bool]]] = lflatten(lll)
    l: List[Tuple[bool, bool]] = lflatten(ll)

    d = compute_f1_from_binary_paired(l)
    return d


def compute_f1_from_binary_paired(l):
    l_inv = [(not b1, not b2) for b1, b2 in l]
    todo = [
        (1, l),
        (0, l_inv)
    ]
    f1_list = []
    cnt_list = []
    acc_list = []
    total = None
    for target_class, paired in todo:
        golds, preds = zip(*paired)
        acc_prec_recall = get_acc_prec_recall(golds, preds)

        prev_total = total
        total = sum([acc_prec_recall[k] for k in ['tp', 'fp', 'tn', 'fn']])
        if prev_total is not None:
            assert total == prev_total

        f1_list.append(acc_prec_recall['f1'])
        acc_list.append(acc_prec_recall['accuracy'])

        cnt = 0
        for g in golds:
            if g:
                cnt += 1

        cnt_list.append(cnt)
    assert_all_equal(acc_list)
    g_total = sum(cnt_list)
    wsum = 0
    for f1, cnt in zip(f1_list, cnt_list):
        weight = cnt / g_total
        wsum += weight * f1
    d = {
        'accuracy': acc_list[0],
        "macro_f1": average(f1_list),
        "weighted": wsum,
        "total": total
    }
    return d


def solve_and_eval(
        solver: PTESolverIF, questions, save_path=None):
    preds: List[PTEPredictionPerQuestion] = apply_solver(solver, questions)

    if save_path is not None:
        save_pte_preds_to_file(preds, save_path)
    d = evaluate(questions, preds)
    return d


def int_return() -> List[int]:
    return [1, 2, 3]


def main():
    solver = PTESolverAllFalse()
    columns_set = set()
    d_d = {}
    for split in sci_ents_all_splits:
        questions = load_scientsbank_split(split)
        d = solve_and_eval(solver, questions)
        columns_set.update(d.keys())
        d_d[split] = d

    columns = list(columns_set)
    head = ["split"] + columns
    rows = [head]
    for split in d_d:
        row = [split]
        for c in columns:
            s = d_d[split][c]
            row.append("{0:.2f}".format(s))

        rows.append(row)

    print_table(rows)


def virtual_pred():
    zero_n = 10318
    one_n = 5945

    golds = [False] * zero_n + [True] * one_n
    preds = [False] * (zero_n + one_n)

    l = list(zip(golds, preds))
    l_inv = [(not b1, not b2) for b1, b2 in l]
    todo = [
        (1, l),
        (0, l_inv)
    ]

    f1_list = []
    cnt_list = []
    acc_list = []

    for target_class, paired in todo:
        golds, preds = zip(*paired)
        acc_prec_recall = get_acc_prec_recall(golds, preds)
        f1_list.append(acc_prec_recall['f1'])
        acc_list.append(acc_prec_recall['accuracy'])

        cnt = 0
        for g in golds:
            if g:
                cnt += 1

        cnt_list.append(cnt)

    assert_all_equal(acc_list)

    total = sum(cnt_list)
    wsum = 0
    for f1, cnt in zip(f1_list, cnt_list):
        weight = cnt / total
        wsum += weight * f1
    d = {
        'accuracy': acc_list[0],
        "macro_f1": average(f1_list),
        "weighted": wsum,
    }
    print(d)


if __name__ == "__main__":
    virtual_pred()