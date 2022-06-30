from typing import List, Tuple

from attribution.attrib_types import TokenScores
from cache import save_to_pickle, load_from_pickle
from contradiction.mnli_ex.load_mnli_ex_data import load_mnli_ex, MNLIExEntry
from contradiction.mnli_ex.mnli_ex_solvers import IdfScorer2, RandomScorer2, ExactMatchScorer
from contradiction.mnli_ex.solver_common import MNLIExSolver
from explain.eval_all import eval_nli_explain

solver_factory_d = {
    'idf2': IdfScorer2,
    'exact_match': ExactMatchScorer,
    'random2': RandomScorer2,
}


def nli_baseline_predict(problems: List[MNLIExEntry], explain_tag, method_name):

    solver: MNLIExSolver = solver_factory_d[method_name]()
    pred_list: List[Tuple[TokenScores, TokenScores]] = solver.explain(problems, explain_tag)
    return pred_list


def get_save_name(method_name, split, target_label):
    save_name = f"mnli_ex_{split}_{target_label}_{method_name}"
    return save_name


def nli_baseline_predict_and_save(split, target_tag, method_name):
    problems = load_mnli_ex(split, target_tag)
    predictions = nli_baseline_predict(problems, "mismatch", method_name)
    save_name = get_save_name(method_name, split, target_tag)
    save_to_pickle(predictions, save_name)


def load_labels_as_tuples(split, label) -> List[Tuple[List[int], List[int]]] :
    e_list = load_mnli_ex(split, label)

    def get(e: MNLIExEntry) -> Tuple[List[int], List[int]]:
        return e.p_indices, e.h_indices

    return list(map(get, e_list))


def run_eval():
    method_name = "random2"
    split = "test"
    target_label = "mismatch"
    run_eval_for(split, target_label, method_name)


def run_eval_for(split, target_label, method_name):
    run_name = get_save_name(method_name, split, target_label)
    predictions = load_from_pickle(run_name)
    gold_list: List[Tuple[List[int], List[int]]] = load_labels_as_tuples(split, target_label)
    scores = eval_nli_explain(predictions, gold_list, False, True)
    print(method_name, scores)


def main():
    split = "test"
    label = "mismatch"
    for method_name in solver_factory_d.keys():
        nli_baseline_predict_and_save(split, label, method_name)
        run_eval_for(split, label, method_name)


if __name__ == "__main__":
    main()