import os
from typing import Iterable, Tuple

from arg.perspectives.load import evidence_gold_dict_str_str
from arg.qck.qrel_helper import get_trec_relevance_judgement
from cpath import data_path


def get_labels() -> Iterable[Tuple[str, str, int]]:
    for query, answers in evidence_gold_dict_str_str().items():
        if not answers:
            print("query has no evidence", query)
        for candidate_id in answers:
            yield query, candidate_id, 1


def main():
    label_itr = list(get_labels())
    l = get_trec_relevance_judgement(label_itr)
    save_path = os.path.join(data_path, "perspective", "evidence_qrel.txt")
    #write_trec_relevance_judgement(l, save_path)


if __name__ == "__main__":
    main()