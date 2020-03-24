from typing import Dict, NamedTuple, List

class MyNamedTuple(NamedTuple):
    doc_id: str
    st: int


def send_queries_passage() -> Dict[str, List[MyNamedTuple]]:
    d = {}
    l = []
    l.append(MyNamedTuple(doc_id="id", st=0))

    d['sstr'] = l
    return d


def other_fn() -> MyNamedTuple:
    r = send_queries_passage()
    return r

