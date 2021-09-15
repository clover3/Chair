from collections import Counter
from typing import NamedTuple

from bs4 import BeautifulSoup

from trec.types import TrecRankedListEntry


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


def get_title(html):
    soup = BeautifulSoup(html, features="html.parser")
    if soup.title is None:
        title = "UNKNOWN"
    else:
        title = soup.title.text
    return str(title)


def enrich(e: TrecRankedListEntry, raw_html_fn, doc_id_to_url) -> WebDocEntry:
    doc_id = e.doc_id
    title = parse_title_from_doc_id(doc_id, raw_html_fn)
    raw_url = doc_id_to_url[e.doc_id]
    url = get_host_name(raw_url)

    return WebDocEntry(e.query_id, e.doc_id, e.rank, e.score, e.run_name, url, title)


def parse_title_from_doc_id(doc_id, raw_html_fn):
    try:
        html = open(raw_html_fn(doc_id), "r", encoding="utf-8")
        title = get_title(html)
    except FileNotFoundError:
        title = "UNKNOWN"
    return title
