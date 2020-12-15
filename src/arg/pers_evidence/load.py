from typing import List, NamedTuple


class PerspectiveClusterWithText(NamedTuple):
    claim_id: str
    claim: str
    perspective_ids: List[str]
    perspective_texts: List[str]
    evidence_ids: List[str]
    evidence_texts: List[str]



