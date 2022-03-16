from collections import defaultdict
from typing import List, Dict

import spacy

from bert_api.segmented_instance.segmented_text import SegmentedText, get_word_level_segmented_text_from_str, \
    seg_to_text
from contradiction.medical_claims.token_tagging.problem_loader import AlamriProblem, load_alamri_problem
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import dict_value_map
from misc_lib import SuccessCounter


class Cursor:
    def __init__(self, tokens):
        self.tokens = tokens
        self.seg_idx = 0
        self.g_idx = 0
        self.in_seg_idx = 0

    def get_char(self):
        return self.tokens[self.seg_idx][self.in_seg_idx]

    def get_seg_idx(self):
        return self.seg_idx

    def move(self):
        self.move_inner()
        while not self.is_end() and self.get_char() == ' ':
            self.move_inner()

    def move_inner(self):
        seg_len = len(self.tokens[self.seg_idx])
        if self.in_seg_idx + 1 < seg_len:
            self.in_seg_idx += 1
        else:
            self.seg_idx += 1
            self.in_seg_idx = 0

        self.g_idx += 1

    def is_end(self):
        return not self.seg_idx < len(self.tokens)


def align_tokens_segmented_text(tokenizer, tokens: List[str], t: SegmentedText) -> Dict[int, List[int]]:
    # Character level alignment
    #
    cursor1 = Cursor(tokens)
    tokens2 = [t.get_seg_text(tokenizer, i) for i in t.enum_seg_idx()]
    cursor2 = Cursor(tokens2)
    overlap = defaultdict(set)
    # overlap[i] includes all segments in t which tokens[i] touches
    while not cursor1.is_end() and not cursor2.is_end():
        char1 = cursor1.get_char().lower()
        char2 = cursor2.get_char().lower()

        seg_idx1 = cursor1.get_seg_idx()
        seg_idx2 = cursor2.get_seg_idx()
        overlap[seg_idx1].add(seg_idx2)

        if char1 == char2:
            cursor1.move()
            cursor2.move()
        else:
            print(tokens)
            print(tokens2)
            print(char1, char2)
            raise Exception()

    def set_to_sorted_list(s):
        l = list(s)
        l.sort()
        return l
    return dict_value_map(set_to_sorted_list, overlap)


def main():
    problems: List[AlamriProblem] = load_alamri_problem()
    tokenizer = get_tokenizer()
    spacy_nlp = spacy.load("en_core_web_sm")

    def enum_sent():
        for p in problems:
            yield p.text1
            yield p.text2

    suc = SuccessCounter()
    for s in enum_sent():
        seg_text: SegmentedText = get_word_level_segmented_text_from_str(tokenizer, s)

        s2 = seg_to_text(tokenizer, seg_text)
        no_space1 = s2.replace(" ", "")

        spacy_tokens = spacy_nlp(s)
        str_tokens = [str(t) for t in spacy_tokens]
        align_mapping = align_tokens_segmented_text(tokenizer, str_tokens, seg_text)
        for idx1, idx2_list in align_mapping.items():
            s1 = str_tokens[idx1].lower()
            s2 = "".join([seg_text.get_seg_text(tokenizer, i) for i in idx2_list])
            if s1 in s2:
                pass
            else:
                print(idx1, idx2_list)
                print(s1, s2)



if __name__ == "__main__":
    main()