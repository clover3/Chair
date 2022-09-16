import json
from typing import List, Iterable, Callable, Dict, Tuple, Set

from alignment import Alignment2D
from alignment.data_structure.matrix_scorer_if import ContributionSummary
from alignment.ists_eval.path_helper import get_ists_2d_save_path
from dataset_specific.ists.parse import ISTSProblem
from dataset_specific.ists.path_helper import load_ists_problems
from misc_lib import two_digit_float
from visualize.html_visual import HtmlVisualizer, Cell


def parse_j(j) -> List[Alignment2D]:
    output = []
    for problem in j:
        problem_id = problem[0]
        contribution_summary = problem[1]
        assert len(contribution_summary) == 1
        table = contribution_summary[0]
        pa = Alignment2D(problem_id, ContributionSummary(table))
        output.append(pa)
    return output


def print_to_html(alignment2d_list: List[Alignment2D], problems: List[ISTSProblem], save_name):
    html = HtmlVisualizer(save_name)

    for alignment2d, problem in zip(alignment2d_list, problems):
        print(alignment2d.problem_id, problem.problem_id)
        assert alignment2d.problem_id == problem.problem_id
        table = alignment2d.contribution.table
        height = len(table)
        width = len(table[0])

        tokens1 = problem.text1.split()
        tokens2 = problem.text2.split()

        assert len(tokens1) == height
        assert len(tokens2) == width

        head = [""] + tokens2
        head_row = list(map(Cell, head))

        out_table = []
        for i, row in enumerate(table):
            score_tokens = list(map(two_digit_float, row))
            norm_score = [int(s*100) for s in row]
            cells = [Cell(tokens1[i])] + [Cell(t, s) for t, s in zip(score_tokens, norm_score)]
            out_table.append(cells)

        html.write_table(out_table, head_row)
        html.write_bar()
        html.f_html.flush()



def main():
    run_name = "nlits_mini"
    genre = "headlines"
    split = "train"
    save_path = get_ists_2d_save_path(genre, split, run_name)
    j = json.load(open(save_path, "r"))
    alignment2d_list: List[Alignment2D] = parse_j(j)
    problems: List[ISTSProblem] = load_ists_problems(genre, split)
    print_to_html(alignment2d_list, problems, run_name + ".html")


if __name__ == "__main__":
    main()