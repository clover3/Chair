import sys
from typing import List, Dict

from arg.perspectives.evaluate import perspective_getter
from arg.perspectives.load import get_all_claim_d, load_evidence_dict, evidence_gold_dict_str_qid
from list_lib import dict_key_map, lmap, foreach
from misc_lib import average
from trec.trec_parse import load_ranked_list_grouped


def main():
    claim_text_d: Dict[int, str] = get_all_claim_d()
    claim_text_d: Dict[str, str] = dict_key_map(str, claim_text_d)
    evi_dict: Dict[str, str] = dict_key_map(str, load_evidence_dict())
    evi_gold_dict: Dict[str, List[int]] = evidence_gold_dict_str_qid()
    print("V2")

    def print_entry(entry):
        evidence_text = evi_dict[entry.doc_id]
        print("[{}] {}: {}".format(entry.rank, entry.doc_id, evidence_text))

    ranked_list_dict = load_ranked_list_grouped(sys.argv[1])
    for query, ranked_list in ranked_list_dict.items():
        print()

        claim_id, perspective_id = query.split("_")
        gold_ids: List[str] = lmap(str, evi_gold_dict[query])
        if not gold_ids:
            print("query {} has no gold".format(query))
            continue
        assert gold_ids
        claim_text = claim_text_d[claim_id]
        perspective_text = perspective_getter(int(perspective_id))

        pos_entries = []
        neg_entries = []
        for entry in ranked_list:
            label = entry.doc_id in gold_ids
            if label:
                pos_entries.append(entry)
            elif entry.rank < 3:
                neg_entries.append(entry)

        if not pos_entries:
            print("gold not in ranked list")
            continue

        num_rel = len(pos_entries)

        correctness = []
        for entry in ranked_list[:num_rel]:
            label = entry.doc_id in gold_ids
            correctness.append(int(label))

        precision = average(correctness)
        if precision > 0.99:
            print("Good")
            continue
        print("precision at {}: {}".format(num_rel, precision))

        print("Claim: ", claim_text)
        print("perspective_text: ", perspective_text)
        print(" < GOLD >")
        foreach(print_entry, pos_entries)
        print(" < False Positive >")
        foreach(print_entry, neg_entries)


if __name__ == "__main__":
    main()