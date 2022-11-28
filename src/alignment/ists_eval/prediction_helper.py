from typing import List, Tuple

from alignment import Alignment2D
from alignment.data_structure.batch_scorer_if import BatchMatrixScorerIF
from alignment.data_structure.matrix_scorer_if import ContributionSummary
from dataset_specific.ists.parse import AlignmentLabelUnit, ALIGN_EQUI, ALIGN_NOALI, ALIGN_SPE1, ALIGN_SPE2, ALIGN_SIMI, \
    ALIGN_REL, iSTSProblemWChunk
from misc_lib import TEL


def convert_2d_to_ists(problems, scores_list: List[Alignment2D],
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


def batch_solve_2d(problems, solver: BatchMatrixScorerIF) -> List[Alignment2D]:
    payload = []
    for p in problems:
        input_per_problem = p.text1.split(), p.text2.split()
        payload.append(input_per_problem)
    batch_output = solver.solve(payload)
    assert len(problems) == len(batch_output)
    predictions: List[Alignment2D] = []
    for p, score_matrix in zip(problems, batch_output):
        predictions.append(Alignment2D(p.problem_id, ContributionSummary(score_matrix)))
    return predictions


def solve_2d_scoring(problems, solver) -> List[Alignment2D]:
    predictions = []
    for p in TEL(problems):
        tokens1 = p.text1.split()
        tokens2 = p.text2.split()
        score_matrix: List[List[float]] = solver.solve(tokens1, tokens2)
        predictions.append(Alignment2D(p.problem_id, ContributionSummary(score_matrix)))
    return predictions


def score_matrix_to_alignment_by_threshold(matrix: List[List[float]], t=0.5) -> List[AlignmentLabelUnit]:
    align_score = 5
    alu_list = []
    aligned_1 = set()
    aligned_2 = set()
    for i1 in range(len(matrix)):
        l2 = len(matrix[i1])
        indices = []
        for i2 in range(l2):
            if matrix[i1][i2] >= t:
                aligned_1.add(i1)
                aligned_2.add(i2)
                indices.append(i2+1)

        if indices:
            alu = AlignmentLabelUnit([i1+1], indices, "dummy1", "dummy2", ["EQUI"], align_score)
            alu_list.append(alu)

    for i1 in range(len(matrix)):
        if i1 not in aligned_1:
            alu = AlignmentLabelUnit([i1+1], [0], "dummy1", "dummy2", ["NOALI"], 0)
            alu_list.append(alu)

    l2 = len(matrix[0])
    for i2 in range(l2):
        if i2 not in aligned_2:
            alu = AlignmentLabelUnit([0], [i2+1], "dummy1", "dummy2", ["NOALI"], 0)
            alu_list.append(alu)

    return alu_list


def score_matrix_to_alignment_by_rank(matrix: List[List[float]], problem) -> Tuple[str, List[AlignmentLabelUnit]]:
    items = []
    n_left = len(matrix)
    n_right = len(matrix[0])
    for i in range(n_left):
        for j in range(n_right):
            item = i, j, matrix[i][j]
            items.append(item)

    items.sort(key=lambda x: x[2], reverse=True)

    aligned_1 = set()
    aligned_2 = set()
    left_items = []
    right_items = []
    labels = []
    for i, j, score in items:
        if i not in aligned_1 and j not in aligned_2 and score > 0:
            left_items.append(i)
            right_items.append(j)
            aligned_1.add(i)
            aligned_2.add(j)
            labels.append([ALIGN_EQUI])

    for i in range(n_left):
        if i not in aligned_1:
            left_items.append(i)
            aligned_1.add(i)
            right_items.append(None)
            labels.append([ALIGN_NOALI])

    for i in range(n_right):
        if i not in aligned_2:
            left_items.append(None)
            right_items.append(i)
            aligned_2.add(i)
            labels.append([ALIGN_NOALI])

    return get_alignment_label_units(left_items, right_items, labels, problem)


def label_to_score(l: List[str]):
    if l[0] == ALIGN_EQUI:
        return 5
    elif l[0] == ALIGN_SPE1 or l[0] == ALIGN_SPE2:
        return 4
    elif l[0] == ALIGN_SIMI:
        return 3
    elif l[0] == ALIGN_REL:
        return 2
    return 0


def get_alignment_label_units(indices1: List[int],
                              indices2: List[int],
                              labels: List[List[str]],
                              problem: iSTSProblemWChunk) -> Tuple[str, List[AlignmentLabelUnit]]:
    n_chunk1 = len(problem.chunks1)
    n_chunk2 = len(problem.chunks2)

    def valid_chunk1(j):
        return indices1[j] is not None

    def valid_chunk2(j):
        return indices2[j] is not None

    assert len(indices1) == len(indices2)
    assert len(indices1) == len(labels)
    n_line = len(indices1)
    aligns: List[AlignmentLabelUnit] = []
    for j in range(n_line):
        if valid_chunk1(j) and valid_chunk2(j):
            i1 = indices1[j]
            i2 = indices2[j]
            score = label_to_score(labels[j])
            u = AlignmentLabelUnit(
                problem.chunk_tokens_ids1[i1], problem.chunk_tokens_ids2[i2],
                problem.chunks1[i1], problem.chunks2[i2],
                labels[j], score)
        elif valid_chunk1(j):
            i1 = indices1[j]
            u = AlignmentLabelUnit(
                problem.chunk_tokens_ids1[i1], [0],
                problem.chunks1[i1], "-not aligned-",
                [ALIGN_NOALI], 0)
        elif valid_chunk2(j):
            i2 = indices2[j]
            u = AlignmentLabelUnit(
                [0], problem.chunk_tokens_ids2[i2],
                "-not aligned-", problem.chunks2[i2],
                [ALIGN_NOALI], 0)
        else:

            assert False
        aligns.append(u)

    aligns.sort(key=AlignmentLabelUnit.sort_key)
    return problem.problem_id, aligns