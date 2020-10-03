from typing import NamedTuple, List, Tuple


class QCKQuery(NamedTuple):
    query_id: str
    text: str


class QCKCandidate(NamedTuple):
    id: str
    text: str


class QCKQueryWToken(NamedTuple):
    query_id: str
    text: str
    tokens: List[str]


class QCKCandidateWToken(NamedTuple):
    id: str
    text: str
    tokens: List[str]


class KnowledgeDocument(NamedTuple):
    doc_id: str
    tokens: List[str]


class KnowledgeDocumentPart(NamedTuple):
    doc_id: str
    passage_idx: int
    start_location: int
    tokens: List[str]

    def getstate(self):
        return self.doc_id, self.passage_idx, self.start_location, self.tokens

    @classmethod
    def from_state(cls, state):
        return KnowledgeDocumentPart(*state)


KD = KnowledgeDocument
KDP = KnowledgeDocumentPart


class QKInstance(NamedTuple):
    query_text: str
    doc_tokens: List[str]
    data_id: int
    is_correct: int


class QKRegressionInstance(NamedTuple):
    query_text: str
    doc_tokens: List[str]
    data_id: int
    score: float


class QCInstance(NamedTuple):
    query_text: str
    candidate_text: str
    data_id: int
    is_correct: int


class QCKInstance(NamedTuple):
    query_text: str
    candidate_text: str
    doc_tokens: List[str]
    data_id: int
    is_correct: int


class QCKInstanceTokenized(NamedTuple):
    query_text: List[str]
    candidate_text: List[str]
    doc_tokens: List[str]
    is_correct: int
    data_id: int


class CKInstance(NamedTuple):
    candidate_text: str
    doc_tokens: List[str]
    data_id: int
    is_correct: int


QKUnit = Tuple[QCKQuery, List[KDP]]


class PayloadAsTokens(NamedTuple):
    passage: List[str]
    text1: List[str]
    text2: List[str]
    data_id: int
    is_correct: int