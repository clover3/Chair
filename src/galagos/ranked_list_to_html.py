import sys
from typing import List, Iterable, Dict

from list_lib import flatten, lmap
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry
from visualize.html_visual import HtmlVisualizer, Cell


def main():
    first_list_path = sys.argv[1]
    dir_path = sys.argv[2]
    save_path = sys.argv[3]
    l: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(first_list_path)

    new_entries: Dict[str, List[TrecRankedListEntry]] = l

    flat_entries: Iterable[TrecRankedListEntry] = flatten(new_entries.values())
    html = HtmlVisualizer(save_path)
    rows = []
    for e in flat_entries:
        ahref = "<a href=\"./{}/{}.html\">{}</a>".format(dir_path, e.doc_id, e.doc_id)
        row = lmap(Cell, [e.query_id, e.rank, e.score, ahref])
        rows.append(row)
    html.write_table(rows)


if __name__ == "__main__":
    main()##
