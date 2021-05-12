import sys

from galagos.types import QueryID
from log_lib import log_variables
from typing import List, Iterable, Callable, Dict, Tuple, Set


def read_queries_at(query_path) -> List[Tuple[QueryID, str]]:
    qid_list = []
    # for line in open(query_path, encoding="utf-8"):
    for line in open(query_path, 'rt', encoding='utf8'):
        qid, q_text = line.split("\t")
        qid_list.append((QueryID(qid), q_text))
    return qid_list


def main():
    q1 = read_queries_at(sys.argv[1])
    q2 = read_queries_at(sys.argv[2])

    print("len(q1)", len(q1))
    print("len(q2)", len(q2))

    q2_d = dict(q2)

    perfect_match = 0
    qid_match = 0
    for query_id, query_text in q1:
        if query_id in q2_d:
            qid_match += 1
            query_text_from2 = q2_d[query_id]
            if query_text.lower() == query_text_from2.lower():
                perfect_match += 1
            else:
                print(query_id)
                print(query_text)
                print(query_text_from2)

    log_variables(perfect_match, qid_match)


if __name__ == "__main__":
    main()