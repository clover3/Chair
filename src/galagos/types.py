from typing import TypeVar, NamedTuple, List, Dict

from base_type import *


class SimpleRankedListEntry(NamedTuple):
    doc_id: str
    rank: int
    score: float


RankedListDict = NewType('RankedListDict', Dict[str, List[SimpleRankedListEntry]])


class GalagoPassageRankEntry(NamedTuple):
    doc_id : str
    st: int
    ed: int
    rank: int
    score: float


GalagoRankEntry = TypeVar('GalagoRankEntry', SimpleRankedListEntry, GalagoPassageRankEntry)

QueryID = NewType('QueryID', str)
QueryResultID = NewType('QueryResultID', str)


class Query(NamedTuple):
    qid: str
    text: str