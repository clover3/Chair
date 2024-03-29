import json
import os
from typing import Dict, Tuple

from cpath import output_path
from data_generator.tokenize_helper import TokenizedText
from data_generator.tokenizer_wo_tf import get_tokenizer
from misc_lib import group_by
from tlm.token_utils import cells_from_tokens
from trec.qrel_parse import load_qrels_structured
from trec.types import QueryID, DocID
from visualize.html_visual import HtmlVisualizer, Cell, normalize


def collect_scores(qrel: Dict[QueryID, Dict[DocID, int]]) -> Dict[Tuple[int, int],
                                                                   Dict[Tuple, Dict]]:

    def get_pair_idx(qid):
        group_no, inner_idx, sent_type, tag_type = qid.split("_")
        group_no = int(group_no)
        inner_idx = int(inner_idx)
        return group_no, inner_idx

    pair_no_grouped = group_by(qrel.keys(), get_pair_idx)
    output = {}
    for pair_no, qids in pair_no_grouped.items():
        per_pair_d = {}
        for qid in qids:
            group_no, inner_idx, sent_type, tag_type = qid.split("_")
            per_pair_d[sent_type, tag_type] = qrel[qid]
        output[pair_no] = per_pair_d
    return output


def print_html(run_name,
               tag_type,
               qrel: Dict[QueryID, Dict[DocID, int]],
               info_d: Dict[int, Dict],
               tokenizer):
    PairNo = Tuple[int, int]
    SentType = Tuple[str, str]
    score_grouped: Dict[PairNo, Dict[SentType, Dict]] = collect_scores(qrel)
    keys = list(score_grouped.keys())
    keys.sort()

    pair_no_index_info = {}
    for data_id, info_e in info_d.items():
        pair_no = info_e['group_no'], info_e['inner_idx']
        pair_no_index_info[pair_no] = info_e

    save_name = "{}_{}.html".format(run_name, tag_type)
    html = HtmlVisualizer(save_name)

    for pair_no in keys:
        print(pair_no)
        local_d = score_grouped[pair_no]
        info_e = pair_no_index_info[pair_no]
        text1 = info_e['text1']
        text2 = info_e['text2']
        t_text1 = TokenizedText.from_text(text1, tokenizer)
        t_text2 = TokenizedText.from_text(text2, tokenizer)
        t_text_d = {
            'prem': t_text1,
            'hypo': t_text2,
        }
        html.write_paragraph("Data no: {} {}".format(pair_no[0], pair_no[1]))
        for sent_type in ["prem", "hypo"]:
            score_d = local_d[sent_type, tag_type]
            t_text = t_text_d[sent_type]
            tokens = t_text.tokens
            doc_ids = [str(i) for i in range(len(tokens))]
            score = [score_d[doc_id]  if doc_id in score_d else 0 for doc_id in doc_ids]
            print(tokens)
            print(score)
            score = normalize(score)
            cells = cells_from_tokens(tokens, score)
            row = [Cell("{}:".format(sent_type))] + cells
            html.multirow_print(row, 20)


# tfrecord/bert_alamri1.pickle
def main():
    data_name = "bert_alamri1"
    tokenizer = get_tokenizer()
    save_dir = os.path.join(output_path, "alamri_annotation1", "tfrecord")
    info_file_path = os.path.join(save_dir, "{}.info".format(data_name))
    info = json.load(open(info_file_path, "r"))
    qrel_name = "worker_J.qrel"

    qrel_path = os.path.join(output_path, "alamri_annotation1", "label",
                             qrel_name)
    qrel = load_qrels_structured(qrel_path)
    for tag_type in ["mismatch", "conflict"]:
        print_html(qrel_name, tag_type, qrel, info, tokenizer)

if __name__ == "__main__":
    main()
