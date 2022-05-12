
from collections import defaultdict
from typing import List, Tuple, NamedTuple

from alignment.data_structure.eval_data_structure import RelatedEvalAnswer, join_a_p
from alignment.data_structure.related_eval_instance import RelatedEvalInstance
from alignment.nli_align_path_helper import load_mnli_rei_problem
from alignment.related.related_answer_data_path_helper import get_related_save_path, load_related_answer_from_path
from bert_api import SegmentedText
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import transpose
from misc_lib import get_second
from models.classic.stopword import load_stopwords
from visualize.html_visual import HtmlVisualizer, Cell, get_bootstrap_include_source


class AlignedSegment(NamedTuple):
    text: SegmentedText
    seg_idx: int
    edges: List[Tuple[int, float]]


def show_alignment(dataset_name, scorer_name, extract_edges, html_save_name):
    save_path = get_related_save_path(dataset_name, scorer_name)
    problem_list: List[RelatedEvalInstance] = load_mnli_rei_problem(dataset_name)
    answer_list: List[RelatedEvalAnswer] = load_related_answer_from_path(save_path)
    tokenizer = get_tokenizer()
    html = HtmlVisualizer(html_save_name, script_include=[get_bootstrap_include_source()])
    html.write_div_open("container")
    stopwords = load_stopwords()
    answer_list = answer_list[:100]
    for answer, problem in join_a_p(answer_list, problem_list):
        html.write_paragraph(problem.problem_id)
        alignment: List[List[float]] = answer.contribution.table
        edges_from_seg1, edges_from_seg2 = extract_edges(alignment)

        seg_inst = problem.seg_instance
        todo = [
            (seg_inst.text1, seg_inst.text2, "hypothesis", edges_from_seg1),
            (seg_inst.text2, seg_inst.text1, "premise", edges_from_seg2),
        ]

        display_max = 3
        for seg_text, other_seg_text, seg_name, edges in todo:
            head = ['Idx', "Word", "Exact match"]
            head.extend(["Edge{}".format(i) for i in range(display_max)])
            head_cells = [Cell(h) for h in head]
            table_tr: List[List] = [head_cells]
            other_seg_words = [other_seg_text.get_seg_text(tokenizer, j) for j in other_seg_text.enum_seg_idx()]
            for seg_idx in seg_text.enum_seg_idx():
                edges_from_here = edges[seg_idx]
                plain_seg_text = seg_text.get_seg_text(tokenizer, seg_idx)
                if plain_seg_text in other_seg_words:
                    exact_match = "Yes"
                else:
                    exact_match = "No"

                row = [str(seg_idx),
                       plain_seg_text,
                       exact_match,
                       ]
                warning = len(edges_from_here) == 0 and seg_name == "hypothesis"
                highlight_score = 50 if warning else 0
                for i in range(display_max):
                    if i < len(edges_from_here):
                        target_idx, score = edges_from_here[i]
                        text = other_seg_text.get_seg_text(tokenizer, target_idx)
                        row.append("{0} ({1:.1f})".format(text, score))
                    else:
                        row.append("-")

                row_cells = [Cell(t, highlight_score, target_color="R") for t in row]
                table_tr.append(row_cells)
            table = transpose(table_tr)

            for row in table:
                row[0].is_head = True

            html.write_headline(seg_name.capitalize())
            head_raw = [c.s for c in table[0]]
            html.write_table(table[1:], head_raw)
            html.write_paragraph("")

        html.write_bar()
    html.write_div_close()


def get_extract_edges_top3():
    top_k= 3
    def extract_edges(alignment: List[List[float]]):
        edges_from_seg1 = defaultdict(list)
        edges_from_seg2 = defaultdict(list)
        for seg_idx1, row in enumerate(alignment):
            candidates: List[Tuple[int, float]] = [(seg_idx2, score) for seg_idx2, score in enumerate(row)]
            candidates.sort(key=get_second, reverse=True)

            for seg_idx2, score in candidates[:top_k]:
                edges_from_seg1[seg_idx1].append((seg_idx2, score))
                edges_from_seg2[seg_idx2].append((seg_idx1, score))
        return edges_from_seg1, edges_from_seg2
    return extract_edges


def main():
    dataset = "train_head"
    model_name = "train_v1_1_2K_linear_9"
    prediction_name = f"pert_{model_name}"
    html_save_name = f"{prediction_name}.html"
    show_alignment(dataset, prediction_name, get_extract_edges_top3(), html_save_name)


if __name__ == "__main__":
    main()