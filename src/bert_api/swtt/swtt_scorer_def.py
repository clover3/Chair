from typing import NamedTuple, List, Tuple

from bert_api.swtt.segmentwise_tokenized_text import SWTTIndex, SegmentwiseTokenizedText
from list_lib import lmap


class SWTTScorerInput(NamedTuple):
    windows_st_ed_list: List[Tuple[SWTTIndex, SWTTIndex]]
    payload_list: List[Tuple[List[str], List[str]]]
    doc: SegmentwiseTokenizedText


class SWTTScorerOutput(NamedTuple):
    windows_st_ed_list: List[Tuple[SWTTIndex, SWTTIndex]]
    scores: List[float]
    doc: SegmentwiseTokenizedText

    def to_json(self):
        windows_st_ed_list = [(st.to_json(), ed.to_json()) for st, ed in self.windows_st_ed_list]
        return {
            'windows_st_ed_list': windows_st_ed_list,
            'scores': self.scores,
            'doc': self.doc.to_json()
        }

    @classmethod
    def from_json(cls, j):
        def conv_st_ed(st_ed):
            st, ed = st_ed
            return SWTTIndex.from_json(st), SWTTIndex.from_json(ed)
        return SWTTScorerOutput(
            lmap(conv_st_ed, j['windows_st_ed_list']),
            j['scores'],
            SegmentwiseTokenizedText.from_json(j['doc'])
        )


    def get_passage(self, passage_idx) -> List[List[str]]:
        st, ed = self.windows_st_ed_list[passage_idx]
        word_tokens_grouped: List[List[str]] = self.doc.get_word_tokens_grouped(st, ed)
        return word_tokens_grouped


class SWTTTokenScorerInput(NamedTuple):
    windows_st_ed_list: List[Tuple[SWTTIndex, SWTTIndex]]
    payload_list: List[Tuple[str, List[List[str]]]]
    doc: SegmentwiseTokenizedText
