from typing import List

from alignment import RelatedEvalAnswer
from alignment.data_structure.batch_scorer_if import BatchMatrixScorerIF
from alignment.data_structure.matrix_scorer_if import ContributionSummary
from dataset_specific.ists.parse import AlignmentLabelUnit
from misc_lib import TEL


def convert_2d_to_ists(problems, scores_list: List[RelatedEvalAnswer],
                       score_matrix_to_alignment_fn):
    predictions = []
    for p, rel_eval_answer in zip(problems, scores_list):
        tokens1 = p.text1.split()
        tokens2 = p.text2.split()
        score_matrix = rel_eval_answer.contribution.table
        alu_list_raw: List[AlignmentLabelUnit] = score_matrix_to_alignment_fn(score_matrix)

        def get_chunk_text(token_ids, tokens):
            if len(token_ids) == 1 and token_ids[0] == 0:
                return "NIL"
            else:
                for i in token_ids:
                    if i == 0:
                        print("WARNING token id==-1 is not expected here")
                return " ".join([tokens[i - 1] for i in token_ids])


        def augment_text(alu: AlignmentLabelUnit):
            chunk1 = get_chunk_text(alu.chunk_token_id1, tokens1)
            chunk2 = get_chunk_text(alu.chunk_token_id2, tokens2)
            return AlignmentLabelUnit(alu.chunk_token_id1, alu.chunk_token_id2,
                                      chunk1, chunk2,
                                      alu.align_types, alu.align_score)

        alu_list = list(map(augment_text, alu_list_raw))
        predictions.append((p.problem_id, alu_list))
    return predictions


def _solve_ists(problems, score_matrix_to_alignment_fn, solver):
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
    return predictions


def batch_solve_2d(problems, solver: BatchMatrixScorerIF) -> List[RelatedEvalAnswer]:
    payload = []
    for p in problems:
        input_per_problem = p.text1.split(), p.text2.split()
        payload.append(input_per_problem)
    batch_output = solver.solve(payload)
    assert len(problems) == len(batch_output)
    predictions: List[RelatedEvalAnswer] = []
    for p, score_matrix in zip(problems, batch_output):
        predictions.append(RelatedEvalAnswer(p.problem_id, ContributionSummary(score_matrix)))
    return predictions


def solve_2d_scoring(problems, solver) -> List[RelatedEvalAnswer]:
    predictions = []
    for p in TEL(problems):
        tokens1 = p.text1.split()
        tokens2 = p.text2.split()
        score_matrix: List[List[float]] = solver.solve(tokens1, tokens2)
        predictions.append(RelatedEvalAnswer(p.problem_id, ContributionSummary(score_matrix)))
    return predictions

