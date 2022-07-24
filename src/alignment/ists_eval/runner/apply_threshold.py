from typing import List, Tuple

from alignment import RelatedEvalAnswer
from alignment.ists_eval.eval_helper import load_ht2d, get_ists_save_path
from alignment.ists_eval.eval_utils import score_matrix_to_alignment_by_threshold, save_ists_predictions
from dataset_specific.ists.parse import AlignmentLabelUnit, ISTSProblem, AlignmentPredictionList
from dataset_specific.ists.path_helper import load_ists_problems
from list_lib import left, pairzip


# AlignmentPredictionList
def augment_problems(ists_raw_preds: AlignmentPredictionList,
                     problems: List[ISTSProblem]) -> AlignmentPredictionList:
    def combine(problem: ISTSProblem, pred: Tuple[str, List[AlignmentLabelUnit]]) -> Tuple[str, List[AlignmentLabelUnit]]:
        tokens1 = problem.text1.split()
        tokens2 = problem.text2.split()
        problem_id, alu_list_raw = pred
        if problem.problem_id != problem_id:
            raise ValueError
        def augment_text(alu: AlignmentLabelUnit):
            chunk1 = " ".join([tokens1[i - 1] for i in alu.chunk_token_id1])
            chunk2 = " ".join([tokens2[i - 1] for i in alu.chunk_token_id2])
            return AlignmentLabelUnit(alu.chunk_token_id1, alu.chunk_token_id2,
                                      chunk1, chunk2,
                                      alu.align_types, alu.align_score)
        alu_list = list(map(augment_text, alu_list_raw))
        return problem.problem_id, alu_list

    assert len(ists_raw_preds) == len(problems)
    return [combine(prob, pred) for prob, pred in zip(problems, ists_raw_preds)]


def apply_threshold_and_save(genre, run_name, split):
    preds: List[RelatedEvalAnswer] = load_ht2d(run_name)

    def convert(p: RelatedEvalAnswer):
        alu_list_raw: List[AlignmentLabelUnit] = score_matrix_to_alignment_by_threshold(p.contribution.table, 0.02)
        return alu_list_raw

    ists_raw_preds: List[Tuple[str, List[AlignmentLabelUnit]]] = pairzip(left(preds), map(convert, preds))
    problems: List[ISTSProblem] = load_ists_problems(genre, split)
    ists_preds = augment_problems(ists_raw_preds, problems)
    save_path = get_ists_save_path(genre, split, run_name)
    save_ists_predictions(ists_preds, save_path)


def main():
    run_name = "coattn"
    genre = "headlines"
    split = "train"
    apply_threshold_and_save(genre, run_name, split)


if __name__ == "__main__":
    main()