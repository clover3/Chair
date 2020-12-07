from typing import List

from arg.qck.decl import QKUnit
from cache import load_from_pickle
from visualize.html_visual import get_collapsible_script, HtmlVisualizer, get_scroll_css, get_collapsible_css


def main():
    # load queires and candidate (from qrel? from BM25 ?)

    # write html
    #   1. Query
    #   2. Doc ID
    #   3. Snippet with most keyword match (BM25 score)
    #   4. scrollable component

    # qk_candidate: List[QKUnit] = load_from_pickle("robust_on_clueweb_qk_candidate")
    qk_candidate: List[QKUnit] = load_from_pickle("robust_on_wiki_qk_candidate")

    # candidates_d = load_candidate_d()
    # save_to_pickle(candidates_d, "candidate_viewer_candidate_d")
    style = [
        get_collapsible_css(),
        get_scroll_css()
    ]
    #
    html = HtmlVisualizer("robust_k_docs_wiki.html",
                          additional_styles=style,
                          )

    for query, k_list in qk_candidate:
        qid = query.query_id
        q_text = query.text

        html.write_div_open()
        html.write_elem("button", "{}: {}".format(qid, q_text),
                        "collapsible",
                        )
        html.write_div_open("content")
        for k in k_list:
            text = " ".join(k.tokens)
            style = "font-size: 13px; padding: 8px;"
            html.write_elem("p", "{}-{}".format(k.doc_id, k.passage_idx), "collapsible", style)
            html.write_div(text, "c_content")
        html.write_div_close()
        html.write_div_close()
    html.write_script(get_collapsible_script())
    html.close()


if __name__ == "__main__":
    main()
