import sys
from typing import List, Iterable, Dict

from evals.trec import load_ranked_list_grouped, write_trec_ranked_list_entry, TrecRankedListEntry
from list_lib import flatten


def main():
    first_list_path = sys.argv[1]
    second_list_path = sys.argv[2]
    save_path = sys.argv[3]
    print("From {} select query that are in {}".format(first_list_path, second_list_path))
    l1: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(first_list_path)
    l2: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(second_list_path)

    new_entries: Dict[str, List[TrecRankedListEntry]] = {}

    for qid in l1:
        if qid in l2:
            new_entries[qid] = l1[qid]

    flat_entries: Iterable[TrecRankedListEntry] = flatten(new_entries.values())
    write_trec_ranked_list_entry(flat_entries, save_path)


if __name__ == "__main__":
    main()
