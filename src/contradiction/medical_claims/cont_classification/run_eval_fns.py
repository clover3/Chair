from typing import List, Iterable, Callable, Dict, Tuple, Set

from contradiction.medical_claims.cont_classification.defs import ContProblem, ContClassificationSolverIF, \
    ContClassificationProbabilityScorer, NOTE_NEG_TYPE1_YS, NOTE_NEG_TYPE1_NO, NOTE_NEG_TYPE2, NOTE_POS_PAIR
from contradiction.medical_claims.cont_classification.path_helper import load_cont_classification_problems, \
    save_predictions, get_prediction_save_path, load_predictions, save_raw_predictions, load_problem_notes
from evals.basic_func import get_acc_prec_recall_i
from trainer_v2.chair_logging import c_log


def run_cont_solver_and_save(solver: ContClassificationSolverIF, run_name, split):
    problems: List[ContProblem] = load_cont_classification_problems(split)
    predictions: List[int] = solver.solve_batch(problems)
    if len(predictions) != len(problems):
        raise IndexError()

    save_predictions(run_name, split, predictions)


def run_cont_prob_solver_and_save(solver: ContClassificationProbabilityScorer, run_name, split):
    c_log.info(f"run_cont_prob_solver_and_save({run_name}, {split})")
    problems: List[ContProblem] = load_cont_classification_problems(split)
    scores: List[float] = solver.solve_batch(problems)
    if len(scores) != len(problems):
        raise IndexError()
    save_raw_predictions(run_name, split, scores)
    predictions = [1 if s >=0.5 else 0 for s in scores]
    save_predictions(run_name, split, predictions)


def do_eval(run_name, split) -> Dict:
    problems: List[ContProblem] = load_cont_classification_problems(split)
    predictions: List[int] = load_predictions(run_name, split)

    if len(predictions) != len(problems):
        raise IndexError()

    labels: List[int] = [p.label for p in problems]

    metrics = get_acc_prec_recall_i(predictions, labels)
    return metrics


def eval_per_category(run_name, split) -> Dict:
    problems: List[ContProblem] = load_cont_classification_problems(split)
    predictions: List[int] = load_predictions(run_name, split)
    notes = load_problem_notes(split)

    if len(predictions) != len(problems):
        raise IndexError()
    pos_pairs = []
    type1_neg_pairs = []
    type2_neg_pairs = []
    for prob, pred in zip(problems, predictions):
        note_text = notes[prob.signature()]
        if note_text == NOTE_NEG_TYPE1_YS or note_text == NOTE_NEG_TYPE1_NO:
            type1_neg_pairs.append((prob, pred))
        elif note_text == NOTE_NEG_TYPE2:
            type2_neg_pairs.append((prob, pred))
        elif note_text == NOTE_POS_PAIR:
            pos_pairs.append((prob, pred))
        else:
            raise KeyError()

    def get_metric_for(pairs: Iterable[Tuple[ContProblem, int]]):
        problems, predictions = zip(*pairs)
        labels: List[int] = [p.label for p in problems]
        base_metrics = get_acc_prec_recall_i(predictions, labels)
        return base_metrics

    return {
        'neg_type1': get_metric_for(type1_neg_pairs),
        'neg_type2': get_metric_for(type2_neg_pairs),
        'pos': get_metric_for(pos_pairs),
    }


def tune_scores(solver: ContClassificationSolverIF, split, target_metric):
    problems: List[ContProblem] = load_cont_classification_problems(split)
    scores: List[float] = list(map(solver.get_raw_score, problems))
    if len(scores) != len(problems):
        raise IndexError()
    labels: List[int] = [p.label for p in problems]

    t_list = range(25, 35)
    max_score = -1
    max_t = 0
    for t in t_list:
        predictions = [1 if s >= t else 0 for s in scores]
        metrics = get_acc_prec_recall_i(predictions, labels)
        s = metrics[target_metric]
        if s > max_score:
            max_score = s
            max_t = t

    print(f"Best of {max_score} at {max_t}")

