from typing import NamedTuple, List, Tuple

from bert_api.swtt.segmentwise_tokenized_text import SWTTIndex, SegmentwiseTokenizedText


class SWTTScorerInput(NamedTuple):
    windows_st_ed_list: List[Tuple[SWTTIndex, SWTTIndex]]
    payload_list: List[Tuple[List[str], List[str]]]
    doc: SegmentwiseTokenizedText


class SWTTScorerOutput(NamedTuple):
    windows_st_ed_list: List[SWTTIndex]
    scores: List[float]
    doc: SegmentwiseTokenizedText