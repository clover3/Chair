import os
from typing import List

from alignment.data_structure.matrix_scorer_if import MatrixScorerIF2
from alignment.ists_eval.eval_utils import save_ists_predictions
from cpath import output_path
from dataset_specific.ists.parse import AlignmentList, parse_label_file, AlignmentLabelUnit
from dataset_specific.ists.path_helper import load_ists_problems
from misc_lib import exist_or_mkdir


def solve_and_save_eval(solver: MatrixScorerIF2, run_name, score_matrix_to_alignment_fn):
    # TODO make genre/split as arguments
    genre = "headlines"
    split = "train"
    problems = load_ists_problems(genre, split)

    predictions = []
    for p in problems:
        tokens1 = p.text1.split()
        tokens2 = p.text2.split()
        score_matrix = solver.solve(tokens1, tokens2)
        alu_list_raw: List[AlignmentLabelUnit] = score_matrix_to_alignment_fn(score_matrix)

        def augment_text(alu: AlignmentLabelUnit):
            chunk1 = " ".join([tokens1[i-1] for i in alu.chunk_token_id1])
            chunk2 = " ".join([tokens2[i-1] for i in alu.chunk_token_id2])
            return AlignmentLabelUnit(alu.chunk_token_id1, alu.chunk_token_id2,
                                      chunk1, chunk2,
                                      alu.align_types, alu.align_score)
        alu_list = list(map(augment_text, alu_list_raw))
        predictions.append((p.problem_id, alu_list))

    save_path = get_ists_save_path(genre, split, run_name)
    save_ists_predictions(predictions, save_path)


def load_ists_predictions(genre, split, run_name) -> AlignmentList:
    return parse_label_file(get_ists_save_path(genre, split, run_name))


def get_ists_save_path(genre, split, run_name) -> str:
    dir_path = os.path.join(output_path, "ists")
    exist_or_mkdir(dir_path)
    return os.path.join(dir_path, f"{genre}.{split}.{run_name}.txt")

