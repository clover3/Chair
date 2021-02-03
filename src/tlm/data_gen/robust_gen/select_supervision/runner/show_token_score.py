import os
from typing import List, Dict

import math

from cache import load_pickle_from
from cpath import at_output_dir
from data_generator.data_parser.robust import load_robust_04_query
from data_generator.job_runner import sydney_working_dir
from data_generator.tokenizer_wo_tf import get_tokenizer
from epath import at_job_man_dir1
from evals.parse import load_qrels_structured
from list_lib import lmap
from misc_lib import tprint, two_digit_float
from tlm.data_gen.robust_gen.select_supervision.token_score import collect_token_scores, DocTokenScore
from visualize.html_visual import HtmlVisualizer, get_tooltip_cell


def main():
    prediction_file_path = at_output_dir("robust", "rob_dense_pred.score")
    info_file_path = at_job_man_dir1("robust_predict_desc_128_step16_info")
    queries: Dict[str, str] = load_robust_04_query("desc")
    tokenizer = get_tokenizer()
    query_token_len_d = {}
    for qid, q_text in queries.items():
        query_token_len_d[qid] = len(tokenizer.tokenize(q_text))
    step_size = 16
    window_size = 128
    out_entries: List[DocTokenScore] = collect_token_scores(info_file_path,
                                                            prediction_file_path,
                                                            query_token_len_d,
                                                            step_size,
                                                            window_size)

    qrel_path = "/home/youngwookim/Downloads/rob04-desc/qrels.rob04.txt"
    judgement_d = load_qrels_structured(qrel_path)

    html = HtmlVisualizer("robust_desc_128_step16.html", use_tooltip=True)

    tprint("loading tokens pickles")
    tokens_d: Dict[str, List[str]] = load_pickle_from(os.path.join(sydney_working_dir, "RobustPredictTokens3", "1"))
    tprint("Now printing")
    n_printed = 0

    def transform(x):
        return 3 * (math.pow(x-0.5, 3) + math.pow(0.5, 3))

    for e in out_entries:
        max_score = e.max_segment_score()
        if max_score < 0.6:
            continue
        n_printed += 1
        if n_printed > 10:
            break
        doc_tokens: List[str] = tokens_d[e.doc_id]
        score_len = len(e.scores)
        judgement: Dict[str, int] = judgement_d[e.query_id]
        label = judgement[e.doc_id]

        if not len(doc_tokens) <= score_len < len(doc_tokens) + window_size:
            print("doc length : ", len(doc_tokens))
            print("score len:", score_len)
            print("doc length +step_size: ", len(doc_tokens)+step_size)
            raise IndexError

        row = []
        q_text = queries[e.query_id]
        html.write_paragraph("qid: " + e.query_id)
        html.write_paragraph("q_text: " + q_text)
        html.write_paragraph("Pred: {0:.2f}".format(max_score))
        html.write_paragraph("Label: {0:.2f}".format(label))

        for idx in range(score_len):
            token = doc_tokens[idx] if idx < len(doc_tokens) else '[-]'

            full_scores = e.full_scores[idx]
            full_score_str = " ".join(lmap(two_digit_float, full_scores))
            score = e.scores[idx]
            normalized_score = transform(score) * 200
            c = get_tooltip_cell(token, full_score_str)
            c.highlight_score = normalized_score
            row.append(c)

        html.multirow_print(row, 16)


if __name__ == "__main__":
    main()
