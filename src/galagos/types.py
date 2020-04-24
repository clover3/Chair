from typing import TypeVar, NamedTuple, List, Dict

from base_type import *


class GalagoDocRankEntry(NamedTuple):
    doc_id: str
    rank: int
    score: float


RankedListDict = NewType('RankedListDict', Dict[str, List[GalagoDocRankEntry]])


class GalagoPassageRankEntry(NamedTuple):
    doc_id : str
    st: int
    ed: int
    rank: int
    score: float


GalagoRankEntry = TypeVar('GalagoRankEntry' , GalagoDocRankEntry, GalagoPassageRankEntry)

QueryID = NewType('QueryID', str)
QueryResultID = NewType('QueryResultID', str)


class Query(NamedTuple):
    qid: str
    text: str