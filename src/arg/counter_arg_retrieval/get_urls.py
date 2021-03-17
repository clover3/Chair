import sys
from typing import List, Iterable, Dict

from cache import save_to_pickle
from clueweb.corpus_reading.doc_id_to_url import get_urls
from list_lib import flatten
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry


def main():
    first_list_path = sys.argv[1]
    l: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(first_list_path)

    new_entries: Dict[str, List[TrecRankedListEntry]] = l

    flat_entries: Iterable[TrecRankedListEntry] = flatten(new_entries.values())
    doc_ids = list(set([e.doc_id for e in flat_entries]))
    urls_d = get_urls(doc_ids)
    save_to_pickle(urls_d, "urls_d")



if __name__ == "__main__":
    main()
