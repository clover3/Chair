from abc import ABC, abstractmethod
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


def get_qk_pair_id(entry) -> Tuple[str, str]:
    return entry['query'].query_id, "{}_{}".format(entry['kdp'].doc_id, entry['kdp'].passage_idx)


def get_qc_pair_id(entry) -> Tuple[str, str]:
    return entry['query'].query_id, entry['candidate'].id


class FormatHandler(ABC):
    @abstractmethod
    def get_pair_id(self, entry):
        pass

    @abstractmethod
    def get_mapping(self):
        pass

    @abstractmethod
    def drop_kdp(self):
        pass


class QCKFormatHandler(FormatHandler):
    def get_pair_id(self, entry):
        return get_qc_pair_id(entry)

    def get_mapping(self):
        return qck_convert_map

    def drop_kdp(self):
        return True


class QCFormatHandler(FormatHandler):
    def get_pair_id(self, entry):
        return get_qc_pair_id(entry)

    def get_mapping(self):
        return qc_convert_map

    def drop_kdp(self):
        return False


class QKFormatHandler(FormatHandler):
    def get_pair_id(self, entry):
        return get_qk_pair_id(entry)

    def get_mapping(self):
        return qk_convert_map

    def drop_kdp(self):
        return False


class QCKLFormatHandler(FormatHandler):
    def get_pair_id(self, entry):
        return get_qc_pair_id(entry)

    def get_mapping(self):
        return qckl_convert_map

    def drop_kdp(self):
        return False


def get_format_handler(input_type):
    if input_type == "qck":
        return QCKFormatHandler()
    elif input_type == "qc":
        return QCFormatHandler()
    elif input_type == "qk":
        return QKFormatHandler()
    elif input_type == "qckl":
        return QCKLFormatHandler()
    else:
        assert False


qck_convert_map = {
        'kdp': KDP,
        'query': QCKQuery,
        'candidate': QCKCandidate
    }
qk_convert_map = {
        'kdp': KDP,
        'query': QCKQuery,
    }
qc_convert_map = {
        'query': QCKQuery,
        'candidate': QCKCandidate,
    }


def parse_kdp_list(*tuple):
    l = list(tuple)
    return list([KDP(*kdp) for kdp in l])



qckl_convert_map = {
        'kdp_list': parse_kdp_list,
        'query': QCKQuery,
        'candidate': QCKCandidate
    }

class QCKOutEntry(NamedTuple):
    logits: List[float]
    query: QCKQuery
    candidate: QCKCandidate
    kdp: KDP

    @classmethod
    def from_dict(cls, d):
        return QCKOutEntry(d['logits'], d['query'], d['candidate'], d['kdp'])