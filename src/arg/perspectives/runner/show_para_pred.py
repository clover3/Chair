from typing import List, Tuple, Dict

from scipy.special import softmax

from arg.pf_common.para_eval import Segment, input_tokens_to_key, split_3segments
from base_type import FilePath, FileName
from cpath import output_path, pjoin
from data_generator.subword_translate import Subword
from data_generator.tokenizer_wo_tf import pretty_tokens
from list_lib import lmap
from misc_lib import group_by
from tlm.estimator_prediction_viewer import EstimatorPredictionViewer
# TODO load predictions
# For each claim-perspective, show suceess cases and failure cases.
from visualize.html_visual import HtmlVisualizer, Cell


def load_prediction(pred_path) -> Dict[str, List[Tuple[str, float, Segment]]]:
    data = EstimatorPredictionViewer(pred_path)

    def parse_entry(entry) -> Tuple[str, float, Segment]:
        input_tokens: Segment = entry.get_tokens('input_ids')
        logits = entry.get_vector("logits")
        probs = softmax(logits)
        key = input_tokens_to_key(input_tokens)
        score = probs[1]

        return key, score, input_tokens

    parsed_data: List[Tuple[str, float, Segment]] = lmap(parse_entry, data)
    grouped: Dict[str, List[Tuple[str, float, Segment]]] = group_by(parsed_data, lambda x: x[0])

    return grouped


def print_file(pred_path):
    grouped = load_prediction(pred_path)
    html_pos = HtmlVisualizer("pc_view_true.html")
    html_neg = HtmlVisualizer("pc_view_false.html")

    item_cnt = 0
    for key in grouped:
        paras: List[Tuple[str, float, Segment]] = grouped[key]

        is_true_arr = list([t[1] > 0.5 for t in paras])
        cnt_true = sum(is_true_arr)
        if cnt_true == len(is_true_arr) or cnt_true == 0:
            continue

        cnt_false = len(is_true_arr) - cnt_true
        idx_false = 0
        idx_true = 0
        item_cnt += 1
        for _, score, tokens in paras:
            is_true = score > 0.5
            html = html_pos if is_true else html_neg
            claim, perspective, paragraph = split_3segments(tokens)
            highlight_terms = set(claim + perspective)
            if is_true:
                html.write_paragraph("{} of {}".format(idx_true, cnt_true))
                idx_true += 1
            else:
                html.write_paragraph("{} of {}".format(idx_false, cnt_false))
                idx_false += 1

            html.write_paragraph("claim : " + pretty_tokens(claim))
            html.write_paragraph("perspective : " + pretty_tokens(perspective))

            def make_cell(subword: Subword):
                if subword in highlight_terms:
                    return Cell(subword, highlight_score=100)
                else:
                    return Cell(subword)

            cells = lmap(make_cell, paragraph)
            html.multirow_print(cells)

        if item_cnt > 100:
            break


if __name__ == "__main__":
    pred_file = FileName("pc_para_D_pred_dev")
    pred_path: FilePath = pjoin(output_path, pred_file)
    print_file(pred_path)
