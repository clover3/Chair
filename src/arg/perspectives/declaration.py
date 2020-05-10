from typing import NamedTuple, List

from arg.pf_common.base import ScoreParagraph


class PerspectiveCandidate(NamedTuple):
    label: str
    cid: str
    pid: str
    claim_text: str
    p_text: str



class ParagraphClaimPersFeature(NamedTuple):
    claim_pers: PerspectiveCandidate
    feature: List[ScoreParagraph]
