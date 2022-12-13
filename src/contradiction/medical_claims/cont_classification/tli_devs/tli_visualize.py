from typing import List

from explain.pairing.run_visualizer.show_cls_probe import NLIVisualize
from visualize.html_visual import Cell


def til_to_table(hypo, tli):
    def prob_to_color(prob) -> str:
        color_score = NLIVisualize.prob_to_color(prob)
        color = "".join([("%02x" % int(v)) for v in color_score])
        return color

    color_array: List[str] = list(map(prob_to_color, tli))
    cell_str_array = list(map(NLIVisualize.get_cell_str, tli))
    row1 = [Cell(t) for t in hypo.split()]
    row2 = []
    row3 = []
    for cell_str, color in zip(cell_str_array, color_array):
        cell = Cell(cell_str, 255, target_color=color)
        row2.append(cell)
        row3.append(Cell(cell_str))
    table = [row1, row2, row3]
    return table