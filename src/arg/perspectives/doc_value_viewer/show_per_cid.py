import os
import sys
from collections import defaultdict
from typing import List, Dict, Tuple

from arg.perspectives.doc_value_viewer.show_doc_value2 import collect_score_per_doc, fetch_score_per_pid, \
    get_score_from_entry, load_claim_d
from arg.perspectives.doc_value_viewer.show_doc_value_per_token import collect_passage_tokens, \
    combine_collect_score
from arg.perspectives.load import get_claim_perspective_id_dict
from arg.perspectives.ppnc.get_doc_value import load_baseline
from cache import load_pickle_from
from cpath import output_path
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import lmap, foreach
from misc_lib import average
from tlm.token_utils import cells_from_tokens
from visualize.html_visual import HtmlVisualizer, Cell


def load_entries(cid):
    save_root = os.path.join(output_path, "cppnc", "cid_grouped")
    save_path = os.path.join(save_root, cid)
    return load_pickle_from(save_path)


def main():
    cid = sys.argv[1]
    print("Loading scores...")
    baseline_cid_grouped = load_baseline("train_baseline")
    gold = get_claim_perspective_id_dict()
    tokenizer = get_tokenizer()
    claim_d = load_claim_d()

    print("Start analyzing")
    html = HtmlVisualizer("cppnc_value_per_token_score_{}.html".format(cid))
    pid_entries_d = load_entries(cid)
    pid_entries_d: Dict[str, List[Dict]] = pid_entries_d
    pid_entries: List[Tuple[str, List[Dict]]] = list(pid_entries_d.items())
    baseline_pid_entries = baseline_cid_grouped[int(cid)]
    baseline_score_d = fetch_score_per_pid(baseline_pid_entries)
    gold_pids = gold[int(cid)]

    ret = collect_score_per_doc(baseline_score_d, get_score_from_entry, gold_pids,
                                                              pid_entries)
    passage_tokens_d = collect_passage_tokens(pid_entries)
    doc_info_d: Dict[int, Tuple[str, int]] = ret[0]
    doc_value_arr: List[List[float]] = ret[1]

    kdp_result_grouped = defaultdict(list)
    for doc_idx, doc_values in enumerate(doc_value_arr):
        doc_id, passage_idx = doc_info_d[doc_idx]
        avg_score = average(doc_values)
        kdp_result = doc_id, passage_idx, avg_score
        kdp_result_grouped[doc_id].append(kdp_result)

    s = "{} : {}".format(cid, claim_d[int(cid)])
    html.write_headline(s)

    scores: List[float] = list([r[2] for r in doc_value_arr])

    foreach(html.write_paragraph, lmap(str, scores))

    for doc_id, kdp_result_list in kdp_result_grouped.items():
        html.write_headline(doc_id)
        tokens, per_token_score = combine_collect_score(tokenizer, doc_id, passage_tokens_d, kdp_result_list)
        str_tokens = tokenizer.convert_ids_to_tokens(tokens)
        row = cells_from_tokens(str_tokens)
        for idx in range(len(str_tokens)):
            score = average(per_token_score[idx])
            norm_score = min(abs(score) * 10000, 100)
            color = "B" if score > 0 else "R"
            row[idx].highlight_score = norm_score
            row[idx].target_color = color

        rows = [row]
        nth = 0
        any_score_found = True
        while any_score_found:
            any_score_found = False
            score_list = []
            for idx in range(len(str_tokens)):
                if nth < len(per_token_score[idx]):
                    score = per_token_score[idx][nth]
                    any_score_found = True
                else:
                    score = "-"
                score_list.append(score)

            def get_cell(score):
                if score == "-":
                    return Cell("-")
                else:
                    # 0.01 -> 100
                    norm_score = min(abs(score) * 20000, 255)
                    color = "B" if score > 0 else "R"
                    return Cell("", highlight_score=norm_score, target_color=color)

            nth += 1
            if any_score_found:
                row = lmap(get_cell, score_list)
                rows.append(row)
        html.multirow_print(rows[0])


if __name__ == "__main__":
    main()