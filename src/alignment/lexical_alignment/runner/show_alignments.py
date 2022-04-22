from collections import defaultdict

from alignment.related.related_answer_data_path_helper import get_related_save_path, load_related_answer_from_path
from alignment.data_structure.eval_data_structure import RelatedEvalAnswer, join_a_p
from alignment.data_structure.related_eval_instance import RelatedEvalInstance
from alignment.nli_align_path_helper import load_mnli_rei_problem
from bert_api import SegmentedText
from bert_api.segmented_instance.seg_instance import SegmentedInstance
from alignment.data_structure.matrix_scorer_if import ContributionSummary, MatrixScorerIF
from typing import List, Iterable, Callable, Dict, Tuple, Set, NamedTuple

from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import transpose
from models.classic.stopword import load_stopwords
from tab_print import print_table
from visualize.html_visual import HtmlVisualizer, Cell, get_bootstrap_include_source


class AlignedSegment(NamedTuple):
    text: SegmentedText
    seg_idx: int
    edges: List[Tuple[int, float]]


def show_alignment(dataset_name, scorer_name):
    save_path = get_related_save_path(dataset_name, scorer_name)
    problem_list: List[RelatedEvalInstance] = load_mnli_rei_problem(dataset_name)
    answer_list: List[RelatedEvalAnswer] = load_related_answer_from_path(save_path)
    tokenizer = get_tokenizer()
    html = HtmlVisualizer("align.html", script_include=[get_bootstrap_include_source()])
    html.write_div_open("container")
    stopwords = load_stopwords()
    g_threshold = 0.5
    answer_list = answer_list[:100]
    for answer, problem in join_a_p(answer_list, problem_list):
        html.write_paragraph(problem.problem_id)
        alignment: List[List[float]] = answer.contribution.table
        edges_from_seg1 = defaultdict(list)
        edges_from_seg2 = defaultdict(list)
        for seg_idx1, row in enumerate(alignment):
            for seg_idx2, score in enumerate(row):
                if score >= g_threshold:
                    edges_from_seg1[seg_idx1].append((seg_idx2, score))
                    edges_from_seg2[seg_idx2].append((seg_idx1, score))

        seg_inst = problem.seg_instance
        todo = [
            (seg_inst.text1, seg_inst.text2, "hypo", edges_from_seg1),
            (seg_inst.text2, seg_inst.text1, "prem", edges_from_seg2),
        ]

        display_max = 3
        for seg_text, other_seg_text, seg_name, edges in todo:
            head = ['idx', seg_name]
            head.extend(["edge{}".format(i) for i in range(display_max)])
            head_cells = [Cell(h) for h in head]
            table_tr: List[List] = [head_cells]
            for seg_idx in seg_text.enum_seg_idx():
                edges_from_here = edges[seg_idx]
                plain_seg_text = seg_text.get_seg_text(tokenizer, seg_idx)

                row = [str(seg_idx),
                       plain_seg_text,
                       ]
                warning = len(edges_from_here) == 0 and seg_name == "hypo"

                def too_many_match():
                    return len(edges_from_here) >= 3 and plain_seg_text not in stopwords

                warning = warning or too_many_match()

                highlight_score = 50 if warning else 0
                for i in range(display_max):
                    if i < len(edges_from_here):
                        target_idx, score = edges_from_here[i]
                        text = other_seg_text.get_seg_text(tokenizer, target_idx)
                        row.append("{0} ({1:.1f})".format(text, score))
                    else:
                        row.append("-")

                n_more = len(edges_from_here) - display_max
                if n_more > 0:
                    row[-1] += " ({} more)".format(n_more)
                row_cells = [Cell(t, highlight_score, target_color="R") for t in row]
                table_tr.append(row_cells)
            table = transpose(table_tr)
            html.write_table(table)
        html.write_bar()
    html.write_div_close()


def main():
    dataset = "train_head"
    show_alignment(dataset, "lexical_v1")


if __name__ == "__main__":
    main()