from typing import List, Dict, Tuple, NamedTuple

from alignment.base_ds import TextPairProblem
from data_generator.tokenize_helper import TokenizedText
from list_lib import index_by_fn, lmap
from misc_lib import two_digit_float
from tlm.token_utils import cells_from_tokens
from trec.types import QRelsDict
from visualize.html_visual import HtmlVisualizer, Cell


def get_qid_pst(pair_no, sent_type, tag_type):
    return f"{pair_no}_{sent_type}_{tag_type}"


class PerPairScores:
    def __init__(self, per_pair_d):
        self.per_pair_d = per_pair_d

    def get_score_as_list(self, tag_type, sent_type):
        score_d: Dict[int, float] = self.per_pair_d[tag_type, sent_type]
        max_idx = max(score_d)
        l = []
        for i in range(max_idx+1):
            l.append(score_d[i])
        return l

    def get_score_as_dict(self, tag_type, sent_type):
        score_d: Dict[int, float] = self.per_pair_d[tag_type, sent_type]
        return score_d


def print_html(save_name,
               tag_type,
               score_grouped: Dict[str, PerPairScores],
               problems: List[TextPairProblem],
               qrel: QRelsDict,
               tokenizer,
               text1_type_name="prem",
               text2_type_name="hypo",
               get_qid=get_qid_pst,
               loc_to_doc_id=str,
               ):
    keys = list(score_grouped.keys())
    keys.sort()

    problems_d: Dict[str, TextPairProblem] = index_by_fn(lambda x: x.problem_id, problems)
    html = HtmlVisualizer(save_name)

    for pair_no in keys:
        per_pair_scores: PerPairScores = score_grouped[pair_no]
        p = problems_d[pair_no]
        t_text1 = TokenizedText.from_text(p.text1, tokenizer)
        t_text2 = TokenizedText.from_text(p.text2, tokenizer)
        t_text_d = {
            text1_type_name: t_text1,
            text2_type_name: t_text2,
        }
        html.write_paragraph("Data no: {}".format(pair_no))
        for sent_type in [text1_type_name, text2_type_name]:
            qid = get_qid(pair_no, sent_type, tag_type)
            try:
                qrel_d: Dict[str, int] = qrel[qid]
            except KeyError as e:
                print(e, "is not in qrel")
                qrel_d = {}
            raw_scores = per_pair_scores.get_score_as_list(sent_type=sent_type, tag_type=tag_type)
            t_text = t_text_d[sent_type]
            tokens = t_text.tokens
            scores = [s * 100 for s in raw_scores]

            score_row = cells_from_tokens(lmap(two_digit_float, raw_scores), scores)
            text_row = cells_from_tokens(tokens, scores)

            def get_qrel_cell(i):
                try:
                    relevant = qrel_d[loc_to_doc_id(i)]
                except KeyError:
                    relevant = 0
                if relevant :
                    return Cell("", highlight_score=100, target_color="G")
                else:
                    return Cell("", highlight_score=0)
            qrel_row = list(map(get_qrel_cell, range(len(tokens))))
            table = [text_row, score_row, qrel_row]
            html.write_paragraph(sent_type)
            html.write_table(table)

