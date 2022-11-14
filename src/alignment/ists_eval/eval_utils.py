from typing import List, Tuple
from xml.etree import ElementTree as ET

from dataset_specific.ists.parse import AlignmentLabelUnit, ALIGN_EQUI, ALIGN_NOALI, iSTSProblemWChunk


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
        if i not in aligned_1 and j not in aligned_2:
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


def save_ists_predictions(predictions: List[Tuple[str, List[AlignmentLabelUnit]]], save_path):
    root = ET.Element('root')
    for problem_id, alu_list in predictions:
        node = ET.SubElement(root, "sentence")
        node.set("id", problem_id)
        node.set("status", "")
        align_node = ET.SubElement(node, "alignment")
        lines = [alu.serialize() for alu in alu_list]
        content = "\n".join(lines)
        content = "\n" + content + "\n"
        align_node.text = content

    s = ET.tostring(root, encoding="unicode")
    s = s.replace("<", "\n<")
    s = s.replace("&lt;", "<")
    s = s.replace("&gt;", ">")
    f = open(save_path, "w")
    f.write(s)


def get_alignment_label_units(indices1: List[int],
                              indices2: List[int],
                              labels: List[List[str]],
                              problem: iSTSProblemWChunk) -> Tuple[str, List[AlignmentLabelUnit]]:
    n_chunk1 = len(problem.chunks1)
    n_chunk2 = len(problem.chunks2)

    def valid_chunk1(j):
        return j < n_chunk1 and indices1[j] is not None

    def valid_chunk2(j):
        return j < n_chunk2 and indices2[j] is not None

    aligns: List[AlignmentLabelUnit] = []
    for j in range(max(n_chunk1, n_chunk2)):
        if valid_chunk1(j) and valid_chunk2(j):
            i1 = indices1[j]
            i2 = indices2[j]
            u = AlignmentLabelUnit(
                problem.chunk_tokens_ids1[i1], problem.chunk_tokens_ids2[i2],
                problem.chunks1[i1], problem.chunks2[i2],
                labels[j], 5)
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
    return problem.problem_id, aligns
