import json
import os
from typing import List

from alignment import RelatedEvalAnswer
from alignment.data_structure.ds_helper import parse_related_eval_answer_from_json
from alignment.data_structure.matrix_scorer_if import MatrixScorerIF2, ContributionSummary
from alignment.ists_eval.eval_utils import save_ists_predictions
from cpath import output_path
from dataset_specific.ists.parse import AlignmentPredictionList, parse_label_file, AlignmentLabelUnit
from dataset_specific.ists.path_helper import load_ists_problems
from misc_lib import exist_or_mkdir, TEL


def solve_as_2d_scores(solver, run_name, genre, split):
    problems = load_ists_problems(genre, split)
    predictions: List[RelatedEvalAnswer] = []
    for p in TEL(problems):
        tokens1 = p.text1.split()
        tokens2 = p.text2.split()
        score_matrix: List[List[float]] = solver.solve(tokens1, tokens2)
        rea = RelatedEvalAnswer(p.problem_id, ContributionSummary(score_matrix))
        predictions.append(rea)
    save_path = get_ists_2d_save_path(genre, split, run_name)
    json.dump(predictions, open(save_path, "w"), indent=True)


def solve_and_save_eval(solver, run_name, score_matrix_to_alignment_fn, genre, split):
    problems = load_ists_problems(genre, split)
    predictions = []
    for p in TEL(problems):
        tokens1 = p.text1.split()
        tokens2 = p.text2.split()
        score_matrix = solver.solve(tokens1, tokens2)
        alu_list_raw: List[AlignmentLabelUnit] = score_matrix_to_alignment_fn(score_matrix)

        def augment_text(alu: AlignmentLabelUnit):
            chunk1 = " ".join([tokens1[i - 1] for i in alu.chunk_token_id1])
            chunk2 = " ".join([tokens2[i - 1] for i in alu.chunk_token_id2])
            return AlignmentLabelUnit(alu.chunk_token_id1, alu.chunk_token_id2,
                                      chunk1, chunk2,
                                      alu.align_types, alu.align_score)

        alu_list = list(map(augment_text, alu_list_raw))
        predictions.append((p.problem_id, alu_list))
    save_path = get_ists_save_path(genre, split, run_name)
    save_ists_predictions(predictions, save_path)


def solve_and_save_eval_ht(solver: MatrixScorerIF2, run_name, score_matrix_to_alignment_fn):
    genre = "headlines"
    split = "train"
    solve_and_save_eval(solver, run_name, score_matrix_to_alignment_fn, genre, split)


def solve_and_save_eval_ht2d(solver: MatrixScorerIF2, run_name):
    genre = "headlines"
    split = "train"
    solve_as_2d_scores(solver, run_name, genre, split)


def load_ht2d(run_name) -> List[RelatedEvalAnswer]:
    genre = "headlines"
    split = "train"
    score_path = get_ists_2d_save_path(genre, split, run_name)
    raw_json = json.load(open(score_path, "r"))
    return parse_related_eval_answer_from_json(raw_json)


def load_ists_predictions(genre, split, run_name) -> AlignmentPredictionList:
    return parse_label_file(get_ists_save_path(genre, split, run_name))


def get_ists_save_path(genre, split, run_name) -> str:
    dir_path = os.path.join(output_path, "ists")
    exist_or_mkdir(dir_path)
    return os.path.join(dir_path, f"{genre}.{split}.{run_name}.txt")


def get_ists_2d_save_path(genre, split, run_name) -> str:
    dir_path = os.path.join(output_path, "ists", "2d")
    exist_or_mkdir(dir_path)
    return os.path.join(dir_path, f"{genre}.{split}.{run_name}.txt")


