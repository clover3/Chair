from typing import NamedTuple, List, Tuple

from data_generator.tokenize_helper import TokenizedText
from list_lib import lmap
from misc_lib import get_duplicate_list


class IntTuple:
    def __init__(self, idx1, idx2):
        self.idx1 = idx1
        self.idx2 = idx2
    
    def get_next_segment(self):
        return IntTuple(self.idx1+1, 0)
    
    def equal(self, idx):
        return self.idx1 == idx.idx1 and self.idx2 == idx.idx2

    def __eq__(self, other):
        return self.equal(other)
    
    def __lt__(self, idx):
        if self.idx1 < idx.idx1:
            return True
        elif self.idx1 == idx.idx1:
            return self.idx2 < idx.idx2
        else:
            return False
    
    def copy(self):
        return IntTuple(self.idx1, self.idx2)

    def __str__(self):
        return "({}, {})".format(self.idx1, self.idx2)

    def to_json(self):
        return (self.idx1, self.idx2)

    @classmethod
    def from_json(cls, j):
        idx1, idx2 = j
        return IntTuple(idx1, idx2)


class SegmentwiseTokenizedText(NamedTuple):
    segments: List[TokenizedText]
    def to_json(self):
        return {
            'segments': [s.to_json() for s in self.segments]
        }

    @classmethod
    def from_json(cls, j):
        return SegmentwiseTokenizedText(lmap(TokenizedText.from_json, j['segments']))

    def get_begin(self) -> IntTuple:
        return IntTuple(0, 0)

    def get_end(self) -> IntTuple:
        return IntTuple(len(self.segments), 0)

    def get_all_segment_lengths(self):
        return [s.get_sb_len() for s in self.segments]

    def is_inside(self, idx: IntTuple):
        return idx < self.get_end()

    def sb_len(self):
        return self.dist(self.get_begin(), self.get_end())

    def dist(self, idx1, idx2):
        rev = False
        if not idx1 < idx2:
            print("Warning idx reversed")
            idx2 = idx1
            rev = True

        ptr: IntTuple = idx1.copy()
        d = 0
        while ptr < idx2:
            to_add = self.segments[ptr.idx1].get_sb_len() - ptr.idx2
            ptr = ptr.get_next_segment()
            d += to_add

        if not rev:
            return d
        else:
            return -d

    def get_window_sb_tokens(self, window: Tuple[IntTuple, IntTuple]) -> List[str]:
        st, ed = window
        ptr: IntTuple = st.copy()
        out_tokens: List[str] = []
        while ptr < ed:
            if ptr.idx1 == ed.idx1:
                to_add: List[str] = self.segments[ptr.idx1].sbword_tokens[ptr.idx2:ed.idx2]
            else:
                to_add: List[str] = self.segments[ptr.idx1].sbword_tokens[ptr.idx2:]

            out_tokens.extend(to_add)
            ptr = ptr.get_next_segment()
        return out_tokens

    def get_word_tokens_grouped(self, st: IntTuple, ed: IntTuple) -> List[List[str]]:
        ptr: IntTuple = st.copy()
        out_tokens = []
        while ptr < ed:
            cur_segment = self.segments[ptr.idx1]
            if cur_segment.sbword_mapping:
                sb_st = cur_segment.sbword_mapping[ptr.idx2]
                if ptr.idx1 == ed.idx1:
                    sb_ed = cur_segment.sbword_mapping[ed.idx2]
                    tokens = cur_segment.tokens[sb_st:sb_ed]
                else:
                    tokens = cur_segment.tokens[sb_st:]

                out_tokens.append(tokens)
            ptr = ptr.get_next_segment()
        return out_tokens

    @classmethod
    def from_text_list(cls, text_list: List[str], tokenizer):
        def inner(text):
            return TokenizedText.from_text(text, tokenizer)

        segments = list(map(inner, text_list))
        return SegmentwiseTokenizedText(segments)

    @classmethod
    def get_duplicate(cls, doc_list: List):
        def para_hash(doc: SegmentwiseTokenizedText):
            return " ".join([" ".join(s.tokens) for s in doc.segments])
        return get_duplicate_list(map(para_hash, doc_list))



SWTTIndex = IntTuple


class PassageSWTT:
    def __init__(self, swtt: SegmentwiseTokenizedText, passage_range: List[Tuple[SWTTIndex, SWTTIndex]]):
        self.swtt = swtt
        self.passage_range = passage_range

    def get_as_word_token_list(self, idx) -> List[List[str]]:
        st, ed = self.passage_range[idx]
        word_tokens_list: List[List[str]] = self.swtt.get_word_tokens_grouped(st, ed)
        return word_tokens_list


class PassageSWTTUnit:
    def __init__(self,
                 swtt: SegmentwiseTokenizedText,
                 passage_range: List[Tuple[SWTTIndex, SWTTIndex]],
                 idx: int):
        self.passage_range = passage_range[idx]
        self.swtt = swtt

    def get_as_subword(self):
        return self.swtt.get_window_sb_tokens(self.passage_range)





