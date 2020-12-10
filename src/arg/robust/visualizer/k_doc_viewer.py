from collections import Counter
from typing import List, Dict

import scipy.special

from arg.qck.decl import QKUnit, qk_convert_map
from arg.qck.prediction_reader import load_combine_info_jsons
from arg.qck.qk_summarize import QKOutEntry
from cache import load_from_pickle
from estimator_helper.output_reader import join_prediction_with_info
from exec_lib import run_func_with_config
from list_lib import lmap
from visualize.html_visual import get_collapsible_script, HtmlVisualizer, get_scroll_css, get_collapsible_css


def load_qk_score_as_dict(config):
    qk_out_entries = load_qk_score(config)

    score_d = {}
    for entry in qk_out_entries:
        key = entry.query.query_id, "{}-{}".format(entry.kdp.doc_id, entry.kdp.passage_idx)
        score_d[key] = scipy.special.softmax(entry.logits)[1]
    return score_d


def load_qk_score(config) -> List[QKOutEntry]:
    info_path = config['info_path']
    passage_score_path = config['pred_path']
    score_type = config['score_type']
    fetch_field_list = ["logits", "input_ids", "data_id"]
    data_id_to_info: Dict = load_combine_info_jsons(info_path, qk_convert_map)
    data: List[Dict] = join_prediction_with_info(passage_score_path,
                                                 data_id_to_info,
                                                 fetch_field_list
                                                 )
    qk_out_entries: List[QKOutEntry] = lmap(QKOutEntry.from_dict2, data)
    return qk_out_entries


def main(config):
    # load queires and candidate (from qrel? from BM25 ?)

    # write html
    #   1. Query
    #   2. Doc ID
    #   3. Snippet with most keyword match (BM25 score)
    #   4. scrollable component

    score_d = load_qk_score_as_dict(config)
    # qk_candidate: List[QKUnit] = load_from_pickle("robust_on_clueweb_qk_candidate")
    qk_candidate: List[QKUnit] = load_from_pickle("robust_on_clueweb_qk_candidate_filtered")
    # qk_candidate: List[QKUnit] = load_from_pickle("robust_on_wiki_qk_candidate")

    # candidates_d = load_candidate_d()
    # save_to_pickle(candidates_d, "candidate_viewer_candidate_d")
    style = [
        get_collapsible_css(),
        get_scroll_css()
    ]
    #
    html = HtmlVisualizer("robust_k_docs_filtered.html",
                          additional_styles=style,
                          )

    for query, k_list in qk_candidate:
        qid = query.query_id
        q_text = query.text
        if not k_list:
            continue

        c = Counter()
        for k in k_list:
            kdp_id = "{}-{}".format(k.doc_id, k.passage_idx)
            score = score_d[qid, kdp_id]
            label = 1 if score > 0.5 else 0
            c[label] += 1

        pos_rate = (c[1] / (c[1] + c[0]))

        html.write_div_open()
        html.write_elem("button", "{0}: {1} ({2:.2f})".format(qid, q_text, pos_rate),
                        "collapsible",
                        )
        html.write_div_open("content")
        for k in k_list:
            kdp_id = "{}-{}".format(k.doc_id, k.passage_idx)
            score = score_d[qid, kdp_id]
            label = score > 0.5
            text = " ".join(k.tokens)
            style = "font-size: 13px; padding: 8px;"
            if label:
                style += " background-color: DarkGreen"
            else:
                style += " background-color: DarkRed"
            html.write_elem("p", "{0} : {1:.2f}".format(kdp_id, score), "collapsible", style)
            html.write_div(text, "c_content")
        html.write_div_close()
        html.write_div_close()
    html.write_script(get_collapsible_script())
    html.close()


if __name__ == "__main__":
    run_func_with_config(main)
