import os
import sys
from typing import List, Iterable, Dict

from arg.counter_arg_retrieval.inspect_html import enrich
from cache import load_from_pickle
from list_lib import flatten, lmap
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry
from visualize.html_visual import HtmlVisualizer, Cell, get_table_head_cell, get_bootstrap_include_source, \
    get_link_highlight_code


def main():
    first_list_path = sys.argv[1]
    dir_path = sys.argv[2]
    save_path = sys.argv[3]
    l: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(first_list_path)

    new_entries: Dict[str, List[TrecRankedListEntry]] = l

    def get_html_path_fn(doc_id):
        return os.path.join(dir_path, "{}.html".format(doc_id))

    doc_id_to_url = load_from_pickle("urls_d")
    flat_entries: Iterable[TrecRankedListEntry] = flatten(new_entries.values())
    entries = [enrich(e, get_html_path_fn, doc_id_to_url) for e in flat_entries]
    html = HtmlVisualizer(save_path, additional_styles=[get_link_highlight_code(), get_bootstrap_include_source()])
    rows = []

    head = [get_table_head_cell("query"),
            get_table_head_cell("rank"),
            get_table_head_cell("score"),
            get_table_head_cell("doc_id"),
            get_table_head_cell("title", 300),
            get_table_head_cell("url"),
            ]

    for e in entries:
        html_path = os.path.join(dir_path, "{}.html".format(e.doc_id))
        ahref = "<a href=\"{}\" target=\"_blank\">{}</a>".format(html_path, e.doc_id)
        elem_list = [e.query_id, e.rank, e.score, ahref, e.title, e.url]
        row = lmap(Cell, elem_list)
        rows.append(row)
    html.write_table_with_class(rows, "table")


if __name__ == "__main__":
    main()
