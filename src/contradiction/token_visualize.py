from typing import List, Dict, Tuple, NamedTuple

from data_generator.tokenize_helper import TokenizedText
from list_lib import index_by_fn, lmap
from misc_lib import two_digit_float
from tlm.token_utils import cells_from_tokens
from trec.types import QRelsDict
from visualize.html_visual import HtmlVisualizer, Cell


class TextPair(NamedTuple):
    problem_id: str
    text1: str
    text2: str


def print_html(save_name,
               tag_type,
               score_grouped: Dict[str, Dict[Tuple[str, str], Dict]],
               problems: List[TextPair],
               qrel: QRelsDict,
               tokenizer):
    SentType = Tuple[str, str]
    keys = list(score_grouped.keys())
    keys.sort()

    problems_d: Dict[str, TextPair] = index_by_fn(lambda x: x.problem_id, problems)
    html = HtmlVisualizer(save_name)

    for pair_no in keys:
        local_d = score_grouped[pair_no]
        p = problems_d[pair_no]
        t_text1 = TokenizedText.from_text(p.text1, tokenizer)
        t_text2 = TokenizedText.from_text(p.text2, tokenizer)
        t_text_d = {
            'prem': t_text1,
            'hypo': t_text2,
        }
        html.write_paragraph("Data no: {}".format(pair_no))
        for sent_type in ["prem", "hypo"]:
            qid = f"{pair_no}_{sent_type}_{tag_type}"
            try:
                qrel_d: Dict[str, int] = qrel[qid]
            except KeyError:
                qrel_d = {}
            score_d = local_d[sent_type, tag_type]
            t_text = t_text_d[sent_type]
            tokens = t_text.tokens
            raw_scores = [score_d[str(i)] for i in range(len(tokens))]
            scores = [s * 100 for s in raw_scores]

            score_row = cells_from_tokens(lmap(two_digit_float, raw_scores), scores)
            text_row = cells_from_tokens(tokens, scores)

            def get_qrel_cell(i):
                try:
                    relevant = qrel_d[str(i)]
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