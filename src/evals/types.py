from typing import List, Dict, Tuple
from typing import NamedTuple

QueryID = str
DocID = str
QRelsFlat = Dict[QueryID, List[Tuple[DocID, int]]]
QRelsDict = Dict[QueryID, Dict[DocID, int]]




class TrecRankedListEntry(NamedTuple):
    query_id: str
    doc_id: str
    rank: int
    score: float
    run_name: str

    def get_doc_id(self):
        return self.doc_id

