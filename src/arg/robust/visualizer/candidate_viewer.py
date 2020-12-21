import os
from typing import List, Dict

from arg.robust.qc_common import to_qck_queries
from cache import load_from_pickle
from cpath import output_path
from data_generator.data_parser.robust import load_robust04_qrels, load_robust04_desc2
from data_generator.data_parser.robust2 import load_bm25_best
from galagos.types import SimpleRankedListEntry
from list_lib import dict_value_map
from trec.trec_parse import load_ranked_list_grouped
from visualize.html_visual import HtmlVisualizer, get_collapsible_script, get_collapsible_css, get_scroll_css


def load_candidate_d():
    candidate_docs: Dict[str, List[SimpleRankedListEntry]] = load_bm25_best()

    def get_doc_id(l: List[SimpleRankedListEntry]):
        return list([e.doc_id for e in l])

    candidate_doc_ids: Dict[str, List[str]] = dict_value_map(get_doc_id, candidate_docs)
    # token_data: Dict[str, List[str]] = load_robust_tokens_for_predict()
    docs = load_from_pickle("robust04_docs_predict")

    out_d = {}
    top_k = 100
    for query_id, doc_id_list in candidate_doc_ids.items():
        new_entries = []
        for doc_id in doc_id_list[:top_k]:
            # tokens = token_data[doc_id]
            content = docs[doc_id]
            new_entries.append((doc_id, content))

        out_d[query_id] = new_entries
    return out_d


def main():
    # load queires and candidate (from qrel? from BM25 ?)

    # write html
    #   1. Query
    #   2. Doc ID
    #   3. Snippet with most keyword match (BM25 score)
    #   4. scrollable component
    ranked_list_path = os.path.join(output_path, "ranked_list", "robust_V_10K_10000.txt")
    bert_ranked_list = load_ranked_list_grouped(ranked_list_path)

    queries: Dict[str, str] = load_robust04_desc2()
    qck_queries = to_qck_queries(queries)
    qrels = load_robust04_qrels()

    candidates_d = load_candidate_d()
    # save_to_pickle(candidates_d, "candidate_viewer_candidate_d")
    # candidates_d = load_from_pickle("candidate_viewer_candidate_d")
    style = [
        get_collapsible_css(),
        get_scroll_css()
    ]
    #
    html = HtmlVisualizer("robust_V_predictions.html",
                          additional_styles=style,
                          )

    def is_perfect(judgement, ranked_list):
        label_list = get_labels(judgement, ranked_list)

        all_relevant = True
        for l in label_list:
            if not l:
                all_relevant = False
            if l:
                if not all_relevant:
                    return False
        return True

    def get_labels(judgement, ranked_list):
        label_list = []
        for e in ranked_list:
            doc_id = e.doc_id
            if doc_id in judgement:
                label = judgement[doc_id]
            else:
                label = 0
            label_list.append(label)
        return label_list

    def p_at_k(judgement, ranked_list, k=10):
        label_list = get_labels(judgement, ranked_list)
        num_correct = sum([1 if label else 0 for label in label_list[:k]])
        return num_correct / k


    for qid in bert_ranked_list :
        if qid in candidates_d:
            if qid not in qrels:
                continue
            judgement = qrels[qid]
            q_text = queries[qid]
            ranked_list = bert_ranked_list[qid]
            if is_perfect(judgement, ranked_list):
                continue

            html.write_div_open()
            text = "{0}: {1} ({2:.2f})".format(qid, q_text, p_at_k(judgement, ranked_list))
            html.write_elem("button", text,
                            "collapsible",
                            )
            html.write_div_open("content")
            doc_text_d = dict(candidates_d[qid])

            for e in ranked_list:
                #tokens = doc_tokens[e.doc_id]
                doc_id = e.doc_id
                if doc_id in judgement:
                    label = judgement[doc_id]
                else:
                    label = 0

                style = "font-size: 13px; padding: 8px;"
                if label:
                    style += " background-color: DarkGreen"
                else:
                    style += " background-color: DarkRed"
                text = "{0}] {1} ({2:.2f})".format(e.rank, doc_id, e.score)
                html.write_elem("p", text, "collapsible", style)
                #text = pretty_tokens(tokens, True)
                doc_text = doc_text_d[doc_id]
                html.write_div(doc_text, "c_content")
            html.write_div_close()
            html.write_div_close()
    html.write_script(get_collapsible_script())
    html.close()


if __name__ == "__main__":
    main()
