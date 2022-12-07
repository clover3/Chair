from collections import Counter
from typing import List, Iterable, Callable, Dict, Tuple, Set

from arg.pers_evidence.common import get_qck_queries
from arg.perspectives.load import evidence_gold_dict_str_qid, load_evidence_dict, splits
from arg.perspectives.kn_tokenizer import KrovetzNLTKTokenizer
from arg.qck.decl import QCKQuery
from arg.qck.filter_qk import text_list_to_lm
from list_lib import lmap


def main():
    for split in splits:
        get_query_lms(split)


def get_query_lms(split) -> Dict[str, Counter]:
    evi_dict: Dict[int, str] = load_evidence_dict()
    tokenzier = KrovetzNLTKTokenizer()
    queries = get_qck_queries(split)
    evi_gold_dict: Dict[str, List[int]] = evidence_gold_dict_str_qid()

    def get_evidence_texts(query: QCKQuery) -> List[str]:
        query_id = query.query_id
        e_ids: List[int] = evi_gold_dict[query_id]
        return list([evi_dict[eid] for eid in e_ids])

    def get_query_lm(query: QCKQuery) -> Counter:
        return text_list_to_lm(tokenzier, get_evidence_texts(query))

    lms = lmap(get_query_lm, queries)
    qids = lmap(QCKQuery.get_id, queries)
    query_lms: Dict[str, Counter] = dict(zip(qids, lms))
    return query_lms


if __name__ == "__main__":
    main()