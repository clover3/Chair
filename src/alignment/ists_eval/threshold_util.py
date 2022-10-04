from typing import List, Tuple

from alignment import Alignment2D
from alignment.ists_eval.eval_helper import load_headline_2d
from alignment.ists_eval.eval_utils import score_matrix_to_alignment_by_threshold, save_ists_predictions
from alignment.ists_eval.f1_calc import calc_f1
from alignment.ists_eval.path_helper import get_ists_save_path

from dataset_specific.ists.parse import AlignmentPredictionList, iSTSProblem, AlignmentLabelUnit
from dataset_specific.ists.path_helper import load_ists_problems, load_ists_label
from list_lib import pairzip, left, lmap
from tab_print import print_table


def augment_problems(ists_raw_preds: AlignmentPredictionList,
                     problems: List[iSTSProblem]) -> AlignmentPredictionList:
    def combine(problem: iSTSProblem, pred: Tuple[str, List[AlignmentLabelUnit]]) -> Tuple[str, List[AlignmentLabelUnit]]:
        tokens1 = problem.text1.split()
        tokens2 = problem.text2.split()
        def get_token(tokens, i):
            if i >= 1:
                return tokens[i-1]
            if i == 0:
                return "-not aligned-"

        problem_id, alu_list_raw = pred
        if problem.problem_id != problem_id:
            raise ValueError
        def augment_text(alu: AlignmentLabelUnit):
            chunk1 = " ".join([get_token(tokens1, i) for i in alu.chunk_token_id1])
            chunk2 = " ".join([get_token(tokens2, i) for i in alu.chunk_token_id2])
            return AlignmentLabelUnit(alu.chunk_token_id1, alu.chunk_token_id2,
                                      chunk1, chunk2,
                                      alu.align_types, alu.align_score)
        alu_list = list(map(augment_text, alu_list_raw))
        return problem.problem_id, alu_list

    assert len(ists_raw_preds) == len(problems)
    return [combine(prob, pred) for prob, pred in zip(problems, ists_raw_preds)]


def apply_threshold_and_save(genre, run_name, split):
    preds: List[Alignment2D] = load_headline_2d(run_name, split)

    def convert(p: Alignment2D):
        alu_list_raw: List[AlignmentLabelUnit] = score_matrix_to_alignment_by_threshold(p.contribution.table, 0.02)
        return alu_list_raw

    ists_raw_preds: List[Tuple[str, List[AlignmentLabelUnit]]] = pairzip(left(preds), map(convert, preds))
    problems: List[iSTSProblem] = load_ists_problems(genre, split)
    ists_preds = augment_problems(ists_raw_preds, problems)
    save_path = get_ists_save_path(genre, split, run_name)
    save_ists_predictions(ists_preds, save_path)


def threshold_tuning(genre, run_name, split):
    if split != "train":
        raise Exception()
    preds: List[Alignment2D] = load_headline_2d(run_name, split)
    mode = ""
    gold: AlignmentPredictionList = load_ists_label(genre, split)

    def eval_for_threshold(threshold):
        def convert_one(p: Alignment2D) -> Tuple[str, List[AlignmentLabelUnit]]:
            alu_list_raw: List[AlignmentLabelUnit] = score_matrix_to_alignment_by_threshold(p.contribution.table, threshold)
            return p.problem_id, alu_list_raw
        ists_raw_preds: List[Tuple[str, List[AlignmentLabelUnit]]] = lmap(convert_one, preds)
        problems: List[iSTSProblem] = load_ists_problems(genre, split)
        ists_preds: AlignmentPredictionList = augment_problems(ists_raw_preds, problems)
        scores = calc_f1(gold, ists_preds, mode)
        return scores

    keys = ["precision", "recall", "f1"]
    head = ["threshold"] + keys
    table = [head]
    for i in range(1, 10):
        t = i * 0.1
        scores = eval_for_threshold(t)
        row = [t] + [scores[k] for k in keys]
        table.append(row)

    print_table(table)

