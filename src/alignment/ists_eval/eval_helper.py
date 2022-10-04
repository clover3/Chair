import json
from typing import Callable, List

from alignment import Alignment2D
from alignment.data_structure.batch_scorer_if import BatchMatrixScorerIF
from alignment.data_structure.ds_helper import parse_related_eval_answer_from_json
from alignment.data_structure.matrix_scorer_if import MatrixScorerIF2
from alignment.ists_eval.eval_utils import save_ists_predictions
from alignment.ists_eval.path_helper import get_ists_save_path, get_ists_2d_save_path
from alignment.ists_eval.prediction_helper import solve_2d_scoring, convert_2d_to_ists, batch_solve_2d
from dataset_specific.ists.parse import AlignmentPredictionList, parse_label_file
from dataset_specific.ists.path_helper import load_ists_problems


def solve_and_save_eval(solver, run_name, score_matrix_to_alignment_fn, genre, split):
    problems = load_ists_problems(genre, split)
    predictions_2d = solve_2d_scoring(problems, solver)
    save_2d_scores(genre, split, run_name, predictions_2d)

    ists_predictions = convert_2d_to_ists(problems, predictions_2d, score_matrix_to_alignment_fn)
    save_path = get_ists_save_path(genre, split, run_name)
    save_ists_predictions(ists_predictions, save_path)


def batch_solve_save_eval(solver: BatchMatrixScorerIF, run_name, score_matrix_to_alignment_fn, genre, split):
    problems = load_ists_problems(genre, split)
    predictions_2d: List[Alignment2D] = batch_solve_2d(problems, solver)
    save_2d_scores(genre, split, run_name, predictions_2d)

    ists_predictions = convert_2d_to_ists(problems, predictions_2d, score_matrix_to_alignment_fn)
    save_path = get_ists_save_path(genre, split, run_name)
    save_ists_predictions(ists_predictions, save_path)


def solve_and_save_eval_mini(solver: MatrixScorerIF2,
                           run_name: str,
                             score_matrix_to_alignment_fn: Callable):
    n_problem = 2
    solve_and_save_eval_part(n_problem, run_name, score_matrix_to_alignment_fn, solver)


def solve_and_save_eval_mini50(solver: MatrixScorerIF2,
                             run_name: str,
                             score_matrix_to_alignment_fn: Callable):
    n_problem = 50
    solve_and_save_eval_part(n_problem, run_name, score_matrix_to_alignment_fn, solver)



def solve_and_save_eval_part(n_problem, run_name, score_matrix_to_alignment_fn, solver):
    genre = "headlines"
    split = "train"
    problems = load_ists_problems(genre, split)
    problems = problems[:n_problem]
    predictions_2d: List[Alignment2D] = solve_2d_scoring(problems, solver)
    save_2d_scores(genre, split, run_name, predictions_2d)
    ists_predictions = convert_2d_to_ists(problems, predictions_2d, score_matrix_to_alignment_fn)
    save_path = get_ists_save_path(genre, split, run_name)
    save_ists_predictions(ists_predictions, save_path)


def load_ists_predictions(genre, split, run_name) -> AlignmentPredictionList:
    return parse_label_file(get_ists_save_path(genre, split, run_name))


def solve_as_2d_scores(solver, run_name, genre, split):
    problems = load_ists_problems(genre, split)
    predictions: List[Alignment2D] = solve_2d_scoring(problems, solver)
    save_2d_scores(genre, split, run_name, predictions)


def save_2d_scores(genre, split, run_name, predictions):
    save_path = get_ists_2d_save_path(genre, split, run_name)
    json.dump(predictions, open(save_path, "w"), indent=1)


def batch_solve_save_eval_headlines_train(solver: BatchMatrixScorerIF,
                                          run_name: str,
                                          score_matrix_to_alignment_fn: Callable):
    genre = "headlines"
    split = "train"
    batch_solve_save_eval(solver, run_name, score_matrix_to_alignment_fn, genre, split)


def solve_and_save_eval_ht(solver: MatrixScorerIF2,
                           run_name: str,
                           score_matrix_to_alignment_fn: Callable):
    genre = "headlines"
    split = "train"
    solve_and_save_eval(solver, run_name, score_matrix_to_alignment_fn, genre, split)


# ht: Headline and Train
def solve_and_save_eval_ht2d(solver: MatrixScorerIF2, run_name):
    genre = "headlines"
    split = "train"
    solve_as_2d_scores(solver, run_name, genre, split)


# ht: Headline and Train
def solve_and_save_eval_headlines_train_2d(solver: MatrixScorerIF2, run_name):
    genre = "headlines"
    split = "train"
    solve_as_2d_scores(solver, run_name, genre, split)


# ht: Headline and Train
def solve_and_save_eval_headlines_2d(solver: MatrixScorerIF2, run_name):
    genre = "headlines"
    solve_as_2d_scores(solver, run_name, genre, "test")
    solve_as_2d_scores(solver, run_name, genre, "train")


def load_headline_2d(run_name, split) -> List[Alignment2D]:
    genre = "headlines"
    score_path = get_ists_2d_save_path(genre, split, run_name)
    raw_json = json.load(open(score_path, "r"))
    return parse_related_eval_answer_from_json(raw_json)
