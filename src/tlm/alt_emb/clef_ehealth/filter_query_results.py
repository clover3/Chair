import os
from typing import Set, Dict, List

from clueweb.sydney_path import get_clueweb12_B13_doc_ids
from cpath import output_path
from galagos.parse import load_galago_ranked_list, FilePath, write_ranked_list_from_s
from galagos.types import GalagoDocRankEntry, RankedListDict
from misc_lib import tprint


def filter_by_doc_id(ranked_list_d: RankedListDict,
                     valid_doc_ids: Set[str]) -> RankedListDict:

    new_d = {}
    for q_id, ranked_list in ranked_list_d.items():
        filtered_entry = [entry for entry in ranked_list if entry.doc_id in valid_doc_ids]
        new_list = []
        for idx, entry in enumerate(filtered_entry):
            new_list.append(GalagoDocRankEntry(doc_id=entry.doc_id, rank=idx+1, score=entry.score))
        new_d[q_id] = new_list

    return new_d


def main():
    ranked_list_path = FilePath(os.path.join(output_path, "eHealth", "bm25.list"))
    save_path = os.path.join(output_path, "eHealth", "bm25_filtered.list")
    tprint("loading ranked list")
    ranked_list: Dict[str, List[GalagoDocRankEntry]] = load_galago_ranked_list(ranked_list_path)

    tprint("loading doc_ids")
    valid_doc_id = get_clueweb12_B13_doc_ids()

    tprint("filtering...")
    new_ranked_list: RankedListDict = filter_by_doc_id(ranked_list, valid_doc_id)

    min_list_len = min([len(ranked_list) for q_id, ranked_list in new_ranked_list.items()])
    print("min_list_len", min_list_len)
    write_ranked_list_from_s(new_ranked_list, save_path)


if __name__ == "__main__":
    main()
