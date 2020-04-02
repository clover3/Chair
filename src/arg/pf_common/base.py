from typing import NamedTuple, List

from data_generator.subword_translate import Subword


# Text Pair Data point
class TPDataPoint(NamedTuple):
    text1: str
    text2: str
    id: str
    label: int


class Paragraph(NamedTuple):
    doc_id: str
    doc_rank: int
    doc_score: float
    tokens: List[str]
    subword_tokens: List[Subword]


class ScoreParagraph(NamedTuple):
    paragraph: Paragraph
    score: float


class ParagraphFeature(NamedTuple):
    datapoint: TPDataPoint
    feature: List[ScoreParagraph]
