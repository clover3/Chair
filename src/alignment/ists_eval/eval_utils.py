from typing import List, Tuple
from xml.etree import ElementTree as ET

from dataset_specific.ists.parse import AlignmentLabelUnit


def score_matrix_to_alignment_by_threshold(matrix: List[List[float]], t=0.5) -> List[AlignmentLabelUnit]:
    align_score = "5"
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
            alu = AlignmentLabelUnit([i1+1], indices, "dummy1", "dummy2", "EQUI", align_score)
            alu_list.append(alu)

    for i1 in range(len(matrix)):
        if i1 not in aligned_1:
            alu = AlignmentLabelUnit([i1+1], [0], "dummy1", "dummy2", "NOALI", "0")
            alu_list.append(alu)

    l2 = len(matrix[0])
    for i2 in range(l2):
        if i2 not in aligned_2:
            alu = AlignmentLabelUnit([0], [i2+1], "dummy1", "dummy2", "NOALI", "0")
            alu_list.append(alu)

    return alu_list


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