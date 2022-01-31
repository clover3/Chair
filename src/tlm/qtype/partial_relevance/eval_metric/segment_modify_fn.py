from typing import Callable, List

import numpy as np

from tlm.qtype.partial_relevance.complement_search_pckg.complement_header import PartialSegment
from tlm.qtype.partial_relevance.complement_search_pckg.complement_search import FuncContentSegJoinPolicy
from tlm.qtype.partial_relevance.eval_data_structure import RelatedEvalAnswer, \
    RelatedEvalInstance, SegmentedInstance
from tlm.qtype.partial_relevance.segmented_text import SegmentedText, get_replaced_segment

DocModFunc = Callable[[SegmentedText, List[float]], SegmentedText]
DocReplaceFunc = Callable[[SegmentedText, List[float], List[int]], SegmentedText]
QueryReplaceFunc = Callable[[SegmentedText, int, List[int]], SegmentedText]


def get_top_k_fn(k) -> DocModFunc:
    def get_top_k_drop_inner(text: SegmentedText, scores: List[float]) -> SegmentedText:
        seg_len = text.get_seg_len()
        drop_len = int(seg_len * k)
        # print("segment scores:", lmap(two_digit_float, scores))
        #
        if len(scores) != seg_len:
            print("Score has {} items while text has {} segments".format(len(scores), seg_len))
        argsorted = np.argsort(scores)
        drop_indices = argsorted[-drop_len:]  # Pick one with highest scores
        new_text = text.get_dropped_text(drop_indices)
        return new_text
    return get_top_k_drop_inner


def get_drop_non_zero() -> DocModFunc:
    def drop_non_zero(text: SegmentedText, scores: List[float]) -> SegmentedText:
        seg_len = text.get_seg_len()
        if len(scores) != seg_len:
            print("Score has {} items while text has {} segments".format(len(scores), seg_len))

        drop_indices = [idx for idx, s in enumerate(scores) if s > 1e-8]
        new_text = text.get_dropped_text(drop_indices)
        return new_text
    return drop_non_zero


def get_drop_zero() -> DocModFunc:
    def drop_zero(text: SegmentedText, scores: List[float]) -> SegmentedText:
        seg_len = text.get_seg_len()
        if len(scores) != seg_len:
            print("Score has {} items while text has {} segments".format(len(scores), seg_len))

        drop_indices = [idx for idx, s in enumerate(scores) if abs(s) < 1e-8]
        new_text = text.get_dropped_text(drop_indices)
        return new_text
    return drop_zero


def assert_float_or_int(v):
    assert type(v) == float or type(v) == int


def get_replace_non_zero() -> DocReplaceFunc:
    def replace_non_zero(text: SegmentedText, scores: List[float], word: List[int]) -> SegmentedText:
        assert_float_or_int(scores[0])
        drop_indices = [idx for idx, s in enumerate(scores) if s > 1e-8]
        return get_replaced_segment(text, drop_indices, word)
    return replace_non_zero


def get_replace_zero() -> DocReplaceFunc:
    def replace_zero(text: SegmentedText, scores: List[float], word: List[int]) -> SegmentedText:
        assert_float_or_int(scores[0])
        drop_indices = [idx for idx, s in enumerate(scores) if abs(s) < 1e-8]
        return get_replaced_segment(text, drop_indices, word)
    return replace_zero


def get_replace_non_target_query() -> QueryReplaceFunc:
    join_policy = FuncContentSegJoinPolicy()
    def query_modify_fn(query: SegmentedText, target_idx, word: List[int]) -> SegmentedText:
        preserve_idx = target_idx
        partial_tokens = PartialSegment.init_one_piece(word)
        new_query: SegmentedText = join_policy.join_tokens(query, partial_tokens, preserve_idx)
        return new_query
    return query_modify_fn


get_replace_not_related: Callable[[], DocReplaceFunc] = get_replace_zero
get_replace_related: Callable[[], DocReplaceFunc] = get_replace_non_zero


# Test TP vs FP (Precision)
# align(qt, dt)
# d = dt + w
# q = qt + w
PairReplaceFunc = Callable[[RelatedEvalAnswer, RelatedEvalInstance, List[int]], SegmentedInstance]


def get_not_related_pair_replace_fn(target_idx) -> PairReplaceFunc:
    doc_modify_fn: DocReplaceFunc = get_replace_not_related()
    query_modify_fn: QueryReplaceFunc = get_replace_non_target_query()
    pair_modify_fn = get_pair_modify_fn(query_modify_fn, doc_modify_fn, target_idx)
    return pair_modify_fn


# Test TN vs FN (Recall)
# align(qt, dt)
# d = d - dt + w
# q = qt + w
def get_related_pair_replace_fn(target_idx) -> PairReplaceFunc:
    doc_modify_fn: DocReplaceFunc = get_replace_related()
    query_modify_fn: QueryReplaceFunc = get_replace_non_target_query()
    pair_modify_fn = get_pair_modify_fn(query_modify_fn, doc_modify_fn, target_idx)
    return pair_modify_fn


def get_pair_modify_fn(query_modify_fn: QueryReplaceFunc,
                       doc_modify_fn,
                       target_seg_idx) -> PairReplaceFunc:
    def pair_modify_fn(answer: RelatedEvalAnswer,
                       problem: RelatedEvalInstance,
                       word: List[int]) -> SegmentedInstance:

        doc_term_scores: List[float] = answer.contribution.table[target_seg_idx]
        new_doc: SegmentedText = doc_modify_fn(problem.seg_instance.text2,
                                               doc_term_scores, word)
        full_query = problem.seg_instance.text1
        new_query: SegmentedText = query_modify_fn(full_query, target_seg_idx, word)
        seg = SegmentedInstance(new_query, new_doc)
        return seg
    return pair_modify_fn