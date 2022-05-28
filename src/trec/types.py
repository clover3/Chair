from typing import List, Dict, Tuple
from typing import NamedTuple

QueryID = str
DocID = str
QRelsFlat = Dict[QueryID, List[Tuple[DocID, int]]]
QRelsDict = Dict[QueryID, Dict[DocID, int]]

QRelsSubtopic = Dict[QueryID, List[Tuple[DocID, str, int]]]


class TrecRankedListEntry(NamedTuple):
    query_id: str
    doc_id: str
    rank: int
    score: float
    run_name: str

    def get_doc_id(self):
        return self.doc_id


class TrecRelevanceJudgementEntry(NamedTuple):
    query_id: str
    doc_id: str
    relevance: int


def load_qrel_as_entries(path) -> Dict[str, List[TrecRelevanceJudgementEntry]]:
    # 101001 0 clueweb12-0001wb-40-32733 0
    q_group = dict()
    for line in open(path, "r"):
        q_id, _, doc_id, score = line.split()
        e = TrecRelevanceJudgementEntry(q_id, doc_id, int(score))
        if q_id not in q_group:
            q_group[q_id] = list()
        q_group[q_id].append(e)
    return q_group
