import json
from typing import List, Dict

import scipy.special
import scipy.special

from contradiction.medical_claims.alamri.pairwise_gen import claim_text_to_info
from contradiction.medical_claims.biobert.voca_common import get_biobert_tokenizer
from contradiction.medical_claims.token_tagging.deletion_score_to_html import make_prediction_summary_str
from data_generator.bert_input_splitter import get_sep_loc
from explain.tf2.deletion_scorer import TokenExEntry, summarize_deletion_score_batch8
from misc_lib import get_second
from tlm.token_utils import cells_from_tokens
from visualize.html_visual import HtmlVisualizer, Cell


def get_neutral_probability(logit):
    return scipy.special.softmax(logit)[1]


def get_contradiction_probability(logit):
    return scipy.special.softmax(logit)[2]

def write_deletion_score_to_html(out_file_name, summarized_table: List[TokenExEntry], info: Dict[int, Dict]):
    text_to_info = claim_text_to_info()
    html = HtmlVisualizer(out_file_name)
    tokenizer = get_biobert_tokenizer()
    num_print = 0
    for entry in summarized_table:
        tokens = tokenizer.convert_ids_to_tokens(entry.input_ids)
        idx_sep1, idx_sep2 = get_sep_loc(entry.input_ids)
        max_change = 0
        max_drop = 0
        cells = cells_from_tokens(tokens)

        drops = []
        for idx in range(len(tokens)):
            if tokens[idx] == "[PAD]":
                break
            if tokens[idx] == '[SEP]':
                continue

            if idx in entry.contribution:
                raw_score = entry.contribution[idx]
                e = idx, raw_score
                drops.append(e)

        drops.sort(key=get_second)
        _, largest_drop = drops[0]

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
                    score = abs(raw_score / largest_drop) * 200
                    color = "B"
                else:
                    score = 0
                    color = "B"
            else:
                score = 150
                color = "Gray"
            cells[idx].highlight_score = score
            cells[idx].target_color = color

        if max_change < 0.05 and False:
            pass
        else:
            # if random.random() < 0.90:
            #     continue
            base_probs = scipy.special.softmax(entry.base_logits)
            info_entry = info[str(entry.data_id[0])]
            claim1_info: Dict = text_to_info[info_entry['text1']]
            claim2_info: Dict = text_to_info[info_entry['text2']]
            question = claim1_info['question']
            assertion1 = claim1_info['assertion']
            assertion2 = claim2_info['assertion']
            original_prediction_summary = make_prediction_summary_str(base_probs)
            html.write_bar()
            html.write_paragraph("Question: {}".format(question))
            html.write_paragraph("Original prediction: " + original_prediction_summary)
            html.write_paragraph("Max drop")

            rows = []
            for idx, score in drops[:5]:
                row = [Cell(str(idx)), Cell(tokens[idx]), Cell(score)]
                rows.append(row)
            html.write_table(rows)

            min_token = tokens[max_drop_idx]
            html.write_paragraph("> \"{}\": {} ".format(min_token, max_drop))
            max_drop_case_prob = scipy.special.softmax(max_drop_case_logit)
            max_drop_prediction_summary = make_prediction_summary_str(max_drop_case_prob)
            html.write_paragraph("> " + max_drop_prediction_summary)
            p = [Cell("Claim1 ({}):".format(assertion1))] + cells[1:idx_sep1]
            h = [Cell("Claim2 ({}):".format(assertion2))] + cells[idx_sep1 + 1:idx_sep2]
            html.write_table([p])
            html.write_table([h])
            num_print += 1

    print("printed {} of {}".format(num_print, len(summarized_table)))



def main():
    dir_path = "C:\\work\\Code\\Chair\\output\\biobert_true_pairs_deletion"
    save_name = "biobert_neutral_normalized"
    info_path = "C:\\work\\Code\\Chair\\output\\alamri_tfrecord\\biobert_true_pairs.info"
    info = json.load(open(info_path, "r", encoding="utf-8"))
    deletion_per_job = 20
    num_jobs = 5
    max_offset = num_jobs * deletion_per_job
    deletion_offset_list = list(range(0, max_offset, deletion_per_job))
    summarized_result = summarize_deletion_score_batch8(dir_path, deletion_per_job,
                                                        deletion_offset_list,
                                                        get_neutral_probability,
                                                        )
    out_file_name = "{}.html".format(save_name)
    write_deletion_score_to_html(out_file_name, summarized_result, info)


if __name__ == "__main__":
    main()