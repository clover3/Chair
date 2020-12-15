from typing import Dict

from arg.perspectives.evaluate import perspective_getter
from arg.perspectives.load import get_all_claim_d, load_evidence_dict, evidence_gold_dict


def main():
    claim_text_d: Dict[int, str] = get_all_claim_d()
    evidence_d = load_evidence_dict()
    evidence_gold = evidence_gold_dict()
    while True:
        s = input()
        cid, pid = s.split("_")
        cid = int(cid)
        pid = int(pid)
        print("Claim: ", claim_text_d[cid])
        print("Perspective: ", perspective_getter(pid))
        key = cid, pid
        e_ids = evidence_gold[key]
        for eid in e_ids:
            print("Evidence: ", evidence_d[eid])


if __name__ == "__main__":
    main()
