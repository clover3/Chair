import os
import sys
from collections import Counter
from typing import List, Iterable, Dict, NamedTuple

from bs4 import BeautifulSoup

from cache import load_from_pickle
from list_lib import flatten, lmap
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry
from visualize.html_visual import HtmlVisualizer, Cell, get_table_head_cell, get_bootstrap_include_source, \
    get_link_highlight_code


class WebDocEntry(NamedTuple):
    query_id: str
    doc_id: str
    rank: int
    score: float
    run_name: str
    url: str
    title: str


def drop_http(url):
    prefix1 = "http://"
    prefix2 = "https://"

    for prefix in [prefix1, prefix2]:
        if url.startswith(prefix):
            return url[len(prefix):]

    raise ValueError(url)


def guess_main_url(url_list):

    new_url_list = []

    for url in url_list:
        try:
            new_url_list.append(drop_http(url))
        except ValueError as e:
            pass

    count = Counter()
    for url in new_url_list:
        head = url.split("/")[0]
        count[head] += 1

    for key, cnt in count.most_common():
        return key
    return "No url"


def get_host_name(url):
    try:
        url = drop_http(url)
        head = url.split("/")[0]
        return head
    except ValueError:
        return url


def enrich(e: TrecRankedListEntry, raw_html_fn, doc_id_to_url) -> WebDocEntry:
    try:
        html = open(raw_html_fn(e.doc_id), "r", encoding="utf-8")
        soup = BeautifulSoup(html)
        title = soup.title.text
        #
        # appear_url = []
        # for link in soup.head.find_all('link'):
        #     appear_url.append(link['href'])
        # for link in soup.head.find_all('script'):
        #     try:
        #         appear_url.append(link['src'])
        #     except:
        #         pass

        raw_url = doc_id_to_url[e.doc_id]
        url = get_host_name(raw_url)
    except Exception as exception:
        print(exception)
        print(e.doc_id)
        url = "UNKNOWN"
        title = "UNKNOWN"

    return WebDocEntry(e.query_id, e.doc_id, e.rank, e.score, e.run_name, url, title)


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
