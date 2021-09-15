import os
from typing import List, Dict

from arg.counter_arg_retrieval.build_dataset.ca_types import CaTopic
from arg.counter_arg_retrieval.build_dataset.resources import load_step2_claims_as_ca_topic
from arg.counter_arg_retrieval.inspect_html import parse_title_from_doc_id, get_host_name
from cache import load_json_cache, dump_to_json
from cpath import output_path
from list_lib import lmap
from misc_lib import exist_or_mkdir, two_digit_float
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry
from visualize.html_visual import HtmlVisualizer, Cell, get_bootstrap_include_source, \
    get_table_head_cell


def main():
    topics: List[CaTopic] = load_step2_claims_as_ca_topic()
    topics_ca_id_index: Dict[str, CaTopic] = {ca.ca_cid: ca for ca in topics}
    urls_d = load_json_cache("ca_urls_d")
    ranking1 = os.path.join(output_path, "ca_building", "run1", "msmarco_ranked_list.txt")
    ranking2 = os.path.join(output_path, "ca_building", "q_res", "q_res_all")
    html_doc_root = os.path.join(output_path, "ca_building", "run1", "html")
    rlg1: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(ranking1)
    rlg2: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(ranking2)
    root_dir = os.path.join(output_path, "ca_building", "run1", "doc_list_html")
    exist_or_mkdir(root_dir)

    def get_html_path(doc_id):
        url = os.path.join(html_doc_root, "{}.html".format(doc_id))
        return url

    for q_id in rlg1:
        print(q_id)
        html_save_path = os.path.join(root_dir, "{}.html".format(q_id))
        exact_match_ranked_list = rlg2[q_id]

        title_cache = "title_d_ca_{}".format(q_id)
        topic = topics_ca_id_index[q_id]
        title_d = load_json_cache(title_cache)
        if title_d is None:
            title_d = {}

        def build_row(e: TrecRankedListEntry) -> List[str]:
            if e.doc_id in title_d:
                title = title_d[e.doc_id]
            else:
                title = parse_title_from_doc_id(e.doc_id, get_html_path)
                title_d[e.doc_id] = title
            msmarco_score = e.score
            em_score = None
            for e2 in exact_match_ranked_list:
                if e2.doc_id == e.doc_id:
                    em_score = e2.score

            url = get_html_path(e.doc_id)
            doc_id_anchor_html = "<a href=\"{}\" target=\"_blank\">{}</a>".format(url, e.doc_id)
            web_url = urls_d[e.doc_id]
            host_name = get_host_name(web_url)
            row = [e.rank,
                   doc_id_anchor_html,
                   two_digit_float(msmarco_score),
                   two_digit_float(em_score),
                   host_name,
                   title,
                   web_url]
            return row

        html = HtmlVisualizer(html_save_path,
                              script_include=[get_bootstrap_include_source()])
        html.write_headline("Claim: {}".format(topic.claim_text), 3)
        html.write_headline("Perspective: {}".format(topic.p_text), 3)
        head = ["rank", "doc_id", "msmarco score", "bm25 score",
                "host",
                get_table_head_cell("title", 30),
                get_table_head_cell("web_url", 10)]

        entries = rlg1[q_id]
        entries = [e for e in entries if e.score > 0.3]
        rows = lmap(build_row, entries)
        rows.sort(key=lambda x: x[3])
        rows = lmap(lambda r: lmap(Cell, r), rows)
        html.write_table_with_class(rows, "table", head)
        if title_d:
            dump_to_json(title_d, title_cache)


if __name__ == "__main__":
    main()