from typing import List

import nltk

from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import flatten
from tlm.qtype.partial_relevance.complement_search_pckg.complement_header import ComplementCandidateGenIF, \
    PartialSegment
from tlm.qtype.partial_relevance.eval_data_structure import SegmentedInstance
from tlm.qtype.partial_relevance.problem_builder import get_word_level_location_w_ids


class ComplementGenBySpanIter(ComplementCandidateGenIF):
    def __init__(self, n_gram_min=1, n_gram_max=3):
        self.tokenizer = get_tokenizer()
        self.n_gram_range = list(range(n_gram_min, n_gram_max+1))

    def get_candidates(self, si: SegmentedInstance, preserve_seg_idx) -> List[PartialSegment]:
        seg2_words = get_word_level_location_w_ids(self.tokenizer, si.text2.tokens_ids)
        candidates: List[PartialSegment] = []
        for n_gram_n in self.n_gram_range:
            ngram_list: List[List[List[int]]] = nltk.ngrams(seg2_words, n_gram_n)
            for n_gram in ngram_list:
                assert len(n_gram) == n_gram_n
                locations = list(flatten(n_gram))
                candidate: List[int] = [si.text2.tokens_ids[i] for i in locations]
                candidates.append(PartialSegment(candidate, 1))
        return candidates

