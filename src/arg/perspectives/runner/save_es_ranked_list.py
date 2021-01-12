import os
from typing import List, Dict, Tuple

from arg.perspectives.eval_helper import get_all_candidate
from arg.perspectives.load import load_claims_for_sub_split
from cpath import output_path
from trec.trec_parse import write_trec_ranked_list_entry, TrecRankedListEntry


def main():
    run_name = "es"
    for split in ["dev", "test"]:
        claims = load_claims_for_sub_split(split)
        candidates_data: List[Tuple[Dict, List[Dict]]] = get_all_candidate(claims)

        flat_entries = []
        for c, candidates in candidates_data:
            assert len(candidates) <= 50
            print(len(candidates))
            query_id = str(c["cId"])

            for rank, e in enumerate(candidates):
                doc_id = str(e['pid'])
                score = e['score']
                entry = TrecRankedListEntry(query_id, doc_id, rank, score, run_name)
                flat_entries.append(entry)

        save_path = os.path.join(output_path, "ranked_list", "pc_es_{}.txt".format(split))
        write_trec_ranked_list_entry(flat_entries, save_path)


if __name__ == "__main__":
    main()