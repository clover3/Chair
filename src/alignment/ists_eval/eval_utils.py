from typing import List, Tuple
from xml.etree import ElementTree as ET

from dataset_specific.ists.parse import AlignmentLabelUnit


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
