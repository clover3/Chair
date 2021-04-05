import json
import sys
from typing import List, Dict

from datastore.interface import load_multiple
from datastore.table_names import RawCluewebDoc
from trec.trec_parse import load_ranked_list
from trec.types import TrecRankedListEntry


def main():
    ranked_list_path = sys.argv[1]
    save_path = sys.argv[2]
    rl: List[TrecRankedListEntry] = load_ranked_list(ranked_list_path)
    doc_ids = list([e.doc_id for e in rl])
    docs_d: Dict[str, List[str]] = {}
    idx = 0
    target_len = 10000
    step = 100
    #

    while idx < target_len:
        print(idx)
        doc_ids_window = doc_ids[idx:idx+step]
        docs_d.update(load_multiple(RawCluewebDoc, doc_ids_window, True))
        idx += step
    print("{} docs_loaded".format(len(docs_d)))
    json.dump(docs_d, open(save_path, "w"))


if __name__ == "__main__":
    main()