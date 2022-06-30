import os
from typing import List, Dict, Tuple

from contradiction.medical_claims.token_tagging.path_helper import get_sbl_qrel_path
from contradiction.medical_claims.token_tagging.problem_loader import AlamriProblem, load_alamri_problem
from cpath import output_path
from data_generator.tokenize_helper import TokenizedText
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import index_by_fn, lmap
from misc_lib import group_by, two_digit_float
from tlm.token_utils import cells_from_tokens
from trec.qrel_parse import load_qrels_structured
from trec.trec_parse import load_ranked_list
from trec.types import TrecRankedListEntry, QRelsDict
from visualize.html_visual import HtmlVisualizer, Cell


def collect_scores(ranked_list: List[TrecRankedListEntry]) -> Dict[str, Dict[Tuple, Dict]]:
    grouped = group_by(ranked_list, lambda x: x.query_id)
    qid_to_score_d = {}
    for qid, entries in grouped.items():
        score_d = {}
        for e in entries:
            score_d[e.doc_id] = e.score
        qid_to_score_d[qid] = score_d

    def get_pair_idx(qid):
        group_no, inner_idx, sent_type, tag_type = qid.split("_")
        group_no = int(group_no)
        inner_idx = int(inner_idx)
        return "{}_{}".format(group_no, inner_idx)

    pair_no_grouped = group_by(qid_to_score_d.keys(), get_pair_idx)
    output = {}
    for pair_no, qids in pair_no_grouped.items():
        per_pair_d = {}
        for qid in qids:
            group_no, inner_idx, sent_type, tag_type = qid.split("_")
            per_pair_d[sent_type, tag_type] = qid_to_score_d[qid]
        output[pair_no] = per_pair_d
    return output


def print_html(run_name,
               tag_type,
               ranked_list,
               problems: List[AlamriProblem],
               qrel: QRelsDict,
               tokenizer):
    SentType = Tuple[str, str]
    score_grouped: Dict[str, Dict[SentType, Dict]] = collect_scores(ranked_list)
    keys = list(score_grouped.keys())
    keys.sort()

    problems_d: Dict[str, AlamriProblem] = index_by_fn(AlamriProblem.get_problem_id, problems)
    save_name = "{}_{}.html".format(run_name, tag_type)
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


# tfrecord/bert_alamri1.pickle
def main():
    tokenizer = get_tokenizer()
    problems: List[AlamriProblem] = load_alamri_problem()

    run_name = "nlits"
    tag_type = "mismatch"
    ranked_list_path = os.path.join(output_path, "alamri_annotation1", "ranked_list",
                                    "{}_{}.txt".format(run_name, tag_type))
    qrel: QRelsDict = load_qrels_structured(get_sbl_qrel_path())
    ranked_list = load_ranked_list(ranked_list_path)
    print_html(run_name, tag_type, ranked_list, problems, qrel, tokenizer)


if __name__ == "__main__":
    main()
