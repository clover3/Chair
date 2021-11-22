from abc import ABC
from typing import List, Tuple

from bert_api.swtt.segmentwise_tokenized_text import SegmentwiseTokenizedText, SWTTIndex


class WindowEnumPolicyI(ABC):
    pass


class WindowEnumPolicy(WindowEnumPolicyI):
    def __init__(self, overlap=0, max_window_per_document=None):
        self.max_window_per_document = max_window_per_document
        if overlap > 0:
            raise NotImplementedError()
        self.overlap = overlap

    def window_enum(self, doc: SegmentwiseTokenizedText, current_limit) -> List[Tuple[SWTTIndex, SWTTIndex]]:
        window_list: List[Tuple[SWTTIndex, SWTTIndex]] = []

        def reached_limit():
            if self.max_window_per_document is not None:
                return len(window_list) >= self.max_window_per_document
            else:
                return False

        cur_st: SWTTIndex = doc.get_begin()
        cur_ed: SWTTIndex = doc.get_begin()

        if doc.sb_len() > 1000 * 300:
            print("Too long document: ", doc.sb_len())

        n_loop = 0
        cur_len = 0
        while cur_st < doc.get_end() and cur_ed < doc.get_end() and not reached_limit():
            next_ed = cur_ed.get_next_segment()
            add_len = doc.dist(cur_ed, next_ed)
            next_len = cur_len + add_len
            if next_len > current_limit:
                if cur_ed.equal(cur_st):
                    cur_ed.idx2 = cur_ed.idx2 + current_limit
                    if not cur_ed.idx2 <= doc.segments[cur_ed.idx1].get_sb_len():
                        print(cur_ed.idx2, doc.segments[cur_ed.idx1].get_sb_len())
                        assert False
                # add to window_list
                window_list.append((cur_st.copy(), cur_ed.copy()))
                cur_st = cur_ed
                cur_len = 0
            else:
                cur_ed = next_ed
                cur_len = next_len

            n_loop += 1

        if cur_st < cur_ed:
            window_list.append((cur_st.copy(), cur_ed.copy()))

        if self.max_window_per_document is not None:
            window_list = window_list[:self.max_window_per_document]

        return window_list


class WindowEnumPolicyMinPop(WindowEnumPolicyI):
    def __init__(self, min_length=100, max_window_per_document=None):
        self.max_window_per_document = max_window_per_document
        self.min_length = min_length

    def window_enum(self, doc: SegmentwiseTokenizedText, current_limit) -> List[Tuple[SWTTIndex, SWTTIndex]]:
        window_list: List[Tuple[SWTTIndex, SWTTIndex]] = []

        def reached_limit():
            if self.max_window_per_document is not None:
                return len(window_list) >= self.max_window_per_document
            else:
                return False

        cur_st: SWTTIndex = doc.get_begin()
        cur_ed: SWTTIndex = doc.get_begin()

        if doc.sb_len() > 1000 * 300:
            print("Too long document: ", doc.sb_len())

        n_loop = 0
        cur_len = 0
        while cur_st < doc.get_end() and cur_ed < doc.get_end() and not reached_limit():
            next_ed = cur_ed.get_next_segment()
            add_len = doc.dist(cur_ed, next_ed)
            next_len = cur_len + add_len
            if next_len > current_limit or cur_len > self.min_length:
                if cur_ed.equal(cur_st):
                    cur_ed.idx2 = cur_ed.idx2 + current_limit
                    if not cur_ed.idx2 <= doc.segments[cur_ed.idx1].get_sb_len():
                        print(cur_ed.idx2, doc.segments[cur_ed.idx1].get_sb_len())
                        assert False
                # add to window_list
                window_list.append((cur_st.copy(), cur_ed.copy()))
                cur_st = cur_ed
                cur_len = 0
            else:
                cur_ed = next_ed
                cur_len = next_len

            n_loop += 1

        if cur_st < cur_ed:
            window_list.append((cur_st.copy(), cur_ed.copy()))

        if self.max_window_per_document is not None:
            window_list = window_list[:self.max_window_per_document]

        return window_list