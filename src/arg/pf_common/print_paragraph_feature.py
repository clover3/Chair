from typing import List

from arg.pf_common.base import ParagraphFeature
from base_type import FilePath
from visualize.html_visual import HtmlVisualizer


def print_paragraph_feature(pf_list: List[ParagraphFeature], out_path: FilePath):
    html = HtmlVisualizer(out_path)
    for pf in pf_list:
        html.write_paragraph("Text 1: " + pf.datapoint.text1)
        html.write_paragraph("Text 2: " + pf.datapoint.text2)
        for f in pf.feature:
            s = " ".join(f.paragraph.tokens)
            html.write_paragraph(s)


    html.close()
