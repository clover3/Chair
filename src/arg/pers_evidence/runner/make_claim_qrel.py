import os
from typing import Iterable, Tuple

from arg.perspectives.load import evidence_gold_dict_str_str
from arg.qck.qrel_helper import get_trec_relevance_judgement
from cpath import data_path
from trec.trec_parse import write_trec_relevance_judgement


def get_labels() -> Iterable[Tuple[str, str, int]]:
    pers_for_claim = {}
    doc_id_for_claim = {}
    for query, answers in evidence_gold_dict_str_str().items():
        claim_id, pers_id = query.split("_")
        if claim_id not in pers_for_claim:
            pers_for_claim[claim_id] = []
            doc_id_for_claim[claim_id] = []

        pers_for_claim[claim_id].append(pers_id)
        for candidate_id in answers:
            doc_id_for_claim[claim_id].append(candidate_id)

    for claim_id in pers_for_claim:
        for pers_id in pers_for_claim[claim_id]:
            query = "{}_{}".format(claim_id, pers_id)
            for doc_id in doc_id_for_claim[claim_id]:
                yield query, doc_id, 1



def main():
    label_itr = list(get_labels())
    l = get_trec_relevance_judgement(label_itr)
    save_path = os.path.join(data_path, "perspective", "claim_evidence_qrel.txt")
    write_trec_relevance_judgement(l, save_path)


if __name__ == "__main__":
    main()