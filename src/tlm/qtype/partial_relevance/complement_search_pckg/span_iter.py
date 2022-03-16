from typing import List

import nltk

from bert_api.segmented_instance.seg_instance import SegmentedInstance
from bert_api.segmented_instance.segmented_text import SegmentedText, get_word_level_location_w_ids
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import flatten
from misc_lib import get_duplicate_list
from tlm.qtype.partial_relevance.complement_search_pckg.complement_header import ComplementCandidateGenIF, \
    PartialSegment


def get_candidates(tokenizer, n_gram_range: List[int], text2: SegmentedText) -> List[PartialSegment]:
    seg2_words = get_word_level_location_w_ids(tokenizer, text2.tokens_ids)
    candidates: List[PartialSegment] = []
    for n_gram_n in n_gram_range:
        ngram_list: List[List[List[int]]] = nltk.ngrams(seg2_words, n_gram_n)
        for n_gram in ngram_list:
            assert len(n_gram) == n_gram_n
            locations = list(flatten(n_gram))
            candidate: List[int] = [text2.tokens_ids[i] for i in locations]
            candidates.append(PartialSegment(candidate, 1))

    def get_sig(ps: PartialSegment) -> str:
        return str(ps.data)

    duplicate_indices = get_duplicate_list(map(get_sig, candidates))
    candidates = [c for idx, c in enumerate(candidates) if idx not in duplicate_indices]
    return candidates


class ComplementGenBySpanIter(ComplementCandidateGenIF):
    def __init__(self, n_gram_min=1, n_gram_max=3):
        self.tokenizer = get_tokenizer()
        self.n_gram_range = list(range(n_gram_min, n_gram_max+1))

    def get_candidates(self, si: SegmentedInstance, preserve_seg_idx) -> List[PartialSegment]:
        return get_candidates(self.tokenizer, self.n_gram_range, si.text2)
