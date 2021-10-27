import os
import re
import urllib.parse
from typing import List, Dict

from arg.counter_arg_retrieval.build_dataset.ca_types import CaTopic
from arg.counter_arg_retrieval.build_dataset.resources import load_step2_claims_as_ca_topic
from arg.counter_arg_retrieval.inspect_html import parse_title_from_doc_id, get_host_name
from cache import load_json_cache
from cpath import output_path
from list_lib import lmap
from misc_lib import exist_or_mkdir, two_digit_float
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry
from visualize.html_visual import HtmlVisualizer, Cell, get_bootstrap_include_source, \
    get_table_head_cell


def write_htmls(rlg1,
                root_dir,
                qid_to_headlines,
                urls_d,
                title_d,
                rel_hint,
                html_doc_root):
    def get_html_path(doc_id):
        url = html_doc_root + "{}.html".format(doc_id)
        return url

    for q_id in rlg1:
        print(q_id)
        doc_list_html_path = os.path.join(root_dir,  "{}.html".format(q_id))
        if title_d is None:
            title_d = {}

        def build_row(e: TrecRankedListEntry) -> List[str]:
            if e.doc_id in title_d:
                title = title_d[e.doc_id]
            else:
                title = parse_title_from_doc_id(e.doc_id, get_html_path)
                title_d[e.doc_id] = title
            msmarco_score = e.score
            url = get_html_path(e.doc_id)
            key = q_id, e.doc_id
            if rel_hint is not None and key in rel_hint:
                st = 3
                slide_len = 30
                repeat = True
                while repeat and st < len(rel_hint[key]):
                    hint_text = rel_hint[key][st:st + slide_len]
                    exclude_chars = ["\"", "’", "'"]
                    repeat = False
                    for c in exclude_chars:
                        if c in hint_text:
                            repeat = True

                    if not repeat:
                        hint_text = re.sub(r" ([/:\.\,\)])", r"\1", hint_text)
                        hint_text = re.sub(r"([\$\(]) ", r"\1", hint_text)
                        print(hint_text)
                    st += slide_len
                url_post_fix = "#:~:text=" + urllib.parse.quote(hint_text)
            else:
                url_post_fix = ""

            url += url_post_fix
            doc_id_anchor_html = "<a href=\"{}\" target=\"_blank\">{}</a>".format(url, e.doc_id)
            web_url = urls_d[e.doc_id]
            host_name = get_host_name(web_url)
            row = [e.rank,
                   doc_id_anchor_html,
                   two_digit_float(msmarco_score),
                   host_name,
                   title,
                   web_url]
            if rel_hint is not None and key in rel_hint:
                row.append(rel_hint[key])
            return row

        html = HtmlVisualizer(doc_list_html_path,
                              script_include=[get_bootstrap_include_source()])

        lines = qid_to_headlines(q_id)
        for line in lines:
            html.write_headline(line, 3)

        head = ["rank", "doc_id", "score",
                "host",
                get_table_head_cell("title", 30),
                get_table_head_cell("web_url", 10)]

        if rel_hint is not None:
            head.append("rel hint")

        entries = rlg1[q_id]
        entries = [e for e in entries if e.score > 0.3]
        rows = lmap(build_row, entries)
        rows.sort(key=lambda x: x[2], reverse=True)
        rows = lmap(lambda r: lmap(Cell, r), rows)
        html.write_table_with_class(rows, "table", head)
        # if title_d:
        #     dump_to_json(title_d, title_cache)


def main():
    topics: List[CaTopic] = load_step2_claims_as_ca_topic()
    topics_ca_id_index: Dict[str, CaTopic] = {ca.ca_cid: ca for ca in topics}
    urls_d = load_json_cache("ca_urls_d")
    ranking1 = os.path.join(output_path, "ca_building", "run1", "msmarco_ranked_list.txt")
    html_doc_root = os.path.join(output_path, "ca_building", "run1", "html")
    rlg1: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(ranking1)
    root_dir = os.path.join(output_path, "ca_building", "run1", "doc_list_html")
    exist_or_mkdir(root_dir)
    title_d_all = {}
    for topic in topics:
        q_id = topic.ca_cid
        title_cache = "title_d_ca_{}".format(q_id)
        title_d = load_json_cache(title_cache)
        title_d_all.update(title_d)


    def qid_to_headlines(qid):
        topic = topics_ca_id_index[qid]
        return [
            "Claim: {}".format(topic.claim_text),
            "Perspective: {}".format(topic.p_text)
        ]
    write_htmls(rlg1,
                root_dir,
                qid_to_headlines,
                urls_d,
                title_d_all,
                html_doc_root)


if __name__ == "__main__":
    main()