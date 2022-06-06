from typing import List, Dict

from contradiction.medical_claims.token_tagging.path_helper import get_save_path2
from contradiction.medical_claims.token_tagging.problem_loader import load_alamri_split, AlamriProblem
from contradiction.medical_claims.token_tagging.query_id_helper import get_query_id
from list_lib import index_by_fn
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry
from visualize.html_visual import HtmlVisualizer, Cell


def simplify_ranked_list(rl: List[TrecRankedListEntry]) -> Dict[str, float]:
    return {e.doc_id: e.score for e in rl}


def visualize(rlg1, rlg2,
              problems: List[AlamriProblem],
              save_name):
    html = HtmlVisualizer(save_name)

    def get_problem_id(p: AlamriProblem) -> str:
        return "{}_{}".format(p.group_no, p.inner_idx)

    problems_d = index_by_fn(get_problem_id, problems)
    for qid in rlg1:
        group_no, inner_idx, sent_name, tag_type = qid.split("_")
        if sent_name == "hypo":
            continue

        try:
            problem = problems_d["{}_{}".format(group_no, inner_idx)]
            for sent_name in ["prem", "hypo"]:
                qid_n = get_query_id(group_no, inner_idx, sent_name, tag_type)
                rl1: List[TrecRankedListEntry] = rlg1[qid_n]
                rl2: List[TrecRankedListEntry] = rlg2[qid_n]

                text = {"prem": problem.text1,
                        "hypo": problem.text2}[sent_name]
                tokens = text.split()

                def get_cells(rl: List[TrecRankedListEntry]) -> List[Cell]:
                    token_score_d = simplify_ranked_list(rl)
                    max_val = max(token_score_d.values())
                    max_val = max(max_val, 1)
                    assert all([s >= -1e-6 for s in token_score_d.values()])

                    def normalize(s):
                        return s / max_val * 100

                    cells = []
                    for idx, token in enumerate(tokens):
                        score = normalize(token_score_d[str(idx)])
                        cells.append(Cell("{0:.2f}".format(score), score))
                    return cells

                rows = [get_cells(rl1), get_cells(rl2)]
                html.write_headline(sent_name)
                html.write_table(rows, tokens)

            html.write_paragraph("")
        except KeyError:
            pass


def main(run1_name, run2_name):
    tag_type = "mismatch"
    rlg_path1 = get_save_path2(run1_name, tag_type)
    rlg_path2 = get_save_path2(run2_name, tag_type)

    visualize(load_ranked_list_grouped(rlg_path1),
              load_ranked_list_grouped(rlg_path2),
              load_alamri_split("dev"),
              "compare_{}_{}.html".format(run1_name, run2_name))


if __name__ == "__main__":
    main("annotator_j", "search1")

