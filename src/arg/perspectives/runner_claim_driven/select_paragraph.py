from typing import Dict, List

from adhoc.clueweb12_B13_termstat import load_clueweb12_B13_termstat
from arg.perspectives.load import get_claims_from_ids, load_dev_claim_ids
from arg.perspectives.select_paragraph_claim import select_paragraph
from cache import save_to_pickle, load_from_pickle


def main():
    docs: Dict[str, List[List[str]]] = load_from_pickle("dev_claim_docs")
    _, clue12_13_df = load_clueweb12_B13_termstat()
    d_ids: List[int] = list(load_dev_claim_ids())
    claims = get_claims_from_ids(d_ids)
    r = select_paragraph(docs, clue12_13_df, claims, "topk")
    save_to_pickle(r, "dev_claim_paras")


if __name__ == "__main__":
    main()


