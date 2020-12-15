import random
import sys
from typing import List, Iterable, Dict

from trec.trec_parse import load_ranked_list_grouped, write_trec_ranked_list_entry, TrecRankedListEntry

from list_lib import flatten


def main():
    input_path = sys.argv[1]
    save_path = sys.argv[2]
    l1: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(input_path)

    new_entries: Dict[str, List[TrecRankedListEntry]] = {}
    run_name = "Random"

    for qid, ranked_list in l1.items():
        raw_ranked_list = []
        for e in ranked_list:
            score = random.random()
            raw_e = (e.query_id, e.doc_id, score)
            raw_ranked_list.append(raw_e)

        raw_ranked_list.sort(key=lambda x: x[2], reverse=True)

        new_ranked_list = []
        for rank, e in enumerate(raw_ranked_list):
            query_id, doc_id, score = e
            e_new = TrecRankedListEntry(query_id, doc_id, rank, score, run_name)
            new_ranked_list.append(e_new)

        new_entries[qid] = new_ranked_list
    flat_entries: Iterable[TrecRankedListEntry] = flatten(new_entries.values())
    write_trec_ranked_list_entry(flat_entries, save_path)


if __name__ == "__main__":
    main()
