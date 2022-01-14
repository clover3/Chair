import random
from typing import List, Dict

import numpy as np
import scipy.special

from contradiction.medical_claims.alamri.pairwise_gen import claim_text_to_info
from contradiction.medical_claims.biobert.voca_common import get_biobert_tokenizer
from data_generator.bert_input_splitter import get_sep_loc
from explain.tf2.deletion_scorer import TokenExEntry
from misc_lib import two_digit_float
from tlm.token_utils import cells_from_tokens
from visualize.html_visual import HtmlVisualizer, Cell


def write_deletion_score_to_html(out_file_name, summarized_table: List[TokenExEntry], info: Dict[int, Dict]):
    text_to_info: Dict[str, Dict] = claim_text_to_info()
    html = HtmlVisualizer(out_file_name)
    tokenizer = get_biobert_tokenizer()
    num_print = 0
    for entry in summarized_table:
        tokens = tokenizer.convert_ids_to_tokens(entry.input_ids)
        idx_sep1, idx_sep2 = get_sep_loc(entry.input_ids)
        max_change = 0
        max_drop = 0
        cells = cells_from_tokens(tokens)
        max_drop_idx = -1
        max_drop_case_logit = None
        for idx in range(len(tokens)):
            if tokens[idx] == "[PAD]":
                break
            if tokens[idx] == '[SEP]':
                continue
            if idx in entry.contribution:
                raw_score = entry.contribution[idx]
                max_change = max(abs(raw_score), max_change)
                if max_drop > raw_score:
                    max_drop = raw_score
                    max_drop_idx = idx
                    max_drop_case_logit = entry.case_logits_d[idx]

                if raw_score < 0:
                    score = abs(raw_score) * 200
                    color = "B"
                else:
                    score = 0
                    color = "B"
            else:
                score = 150
                color = "Gray"
            cells[idx].highlight_score = score
            cells[idx].target_color = color

        if max_change < 0.05:
            pass
        else:
            if random.random() < 0.90:
                continue
            base_probs = scipy.special.softmax(entry.base_logits)
            info_entry = info[str(entry.data_id[0])]
            claim1_info: Dict = text_to_info[info_entry['text1']]
            claim2_info: Dict = text_to_info[info_entry['text2']]
            question = claim1_info['question']
            assertion1 = claim1_info['assertion']
            assertion2 = claim2_info['assertion']
            html.write_paragraph("Question: {}".format(question))
            original_prediction_summary = make_nli_prediction_summary_str(base_probs)
            html.write_bar()
            html.write_paragraph("Original prediction: " + original_prediction_summary)
            html.write_paragraph("Question: {}".format(question))
            html.write_paragraph("Max drop")
            min_token = tokens[max_drop_idx]
            html.write_paragraph("> \"{}\": {} ".format(min_token, max_drop))
            max_drop_case_prob = scipy.special.softmax(max_drop_case_logit)
            max_drop_prediction_summary = make_nli_prediction_summary_str(max_drop_case_prob)
            html.write_paragraph("> " + max_drop_prediction_summary)
            p = [Cell("Claim1 ({}):".format(assertion1))] + cells[1:idx_sep1]
            h = [Cell("Claim2 ({}):".format(assertion2))] + cells[idx_sep1 + 1:idx_sep2]
            html.write_table([p])
            html.write_table([h])
            num_print += 1

    print("printed {} of {}".format(num_print, len(summarized_table)))


def make_nli_prediction_summary_str(base_probs):
    pred = np.argmax(base_probs)
    orignal_prediction_str = ['entailment', 'neutral', 'contradiction'][pred]
    original_prediction_summary = "{} ({}, {}, {})".format(orignal_prediction_str,
                                                           two_digit_float(base_probs[0]),
                                                           two_digit_float(base_probs[1]),
                                                           two_digit_float(base_probs[2]),
                                                           )
    return original_prediction_summary