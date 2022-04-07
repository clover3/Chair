import random
from typing import Callable, List

import numpy as np

from bert_api.segmented_instance.seg_instance import SegmentedInstance
from bert_api.segmented_instance.segmented_text import SegmentedText, get_replaced_segment
from misc_lib import average
from tlm.qtype.partial_relevance.complement_search_pckg.complement_header import PartialSegment
from tlm.qtype.partial_relevance.complement_search_pckg.complement_search import FuncContentSegJoinPolicy
from contradiction.alignment.data_structure.eval_data_structure import RelatedBinaryAnswer
from tlm.qtype.partial_relevance.related_eval_instance import RelatedEvalInstance
from tlm.qtype.partial_relevance.eval_metric.ep_common import DropSamplePolicyIF, ReplaceSamplePolicyIF

DocModFuncR = Callable[[SegmentedText, List[float]], SegmentedText]
DocModFuncB = Callable[[SegmentedText, List[int]], SegmentedText]

DocReplaceFuncR = Callable[[SegmentedText, List[float], List[int]], SegmentedText]
DocReplaceFuncB = Callable[[SegmentedText, List[int], List[int]], SegmentedText]

QueryReplaceFunc = Callable[[SegmentedText, int, List[int]], SegmentedText]


def get_top_k_fn(k) -> DocModFuncR:
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


def get_drop_non_zero() -> DocModFuncB:
    def drop_non_zero(text: SegmentedText, scores: List[int]) -> SegmentedText:
        seg_len = text.get_seg_len()
        if len(scores) != seg_len:
            print("Score has {} items while text has {} segments".format(len(scores), seg_len))

        for s in scores:
            assert type(s) == int

        drop_indices = [idx for idx, s in enumerate(scores) if s == 1]
        new_text = text.get_dropped_text(drop_indices)
        return new_text
    return drop_non_zero


def get_drop_zero() -> DocModFuncB:
    def drop_zero(text: SegmentedText, scores: List[int]) -> SegmentedText:
        seg_len = text.get_seg_len()
        if len(scores) != seg_len:
            print("Score has {} items while text has {} segments".format(len(scores), seg_len))

        drop_indices = [idx for idx, s in enumerate(scores) if s == 0]
        new_text = text.get_dropped_text(drop_indices)
        return new_text
    return drop_zero


def assert_float_or_int(v):
    assert type(v) == float or type(v) == int


def get_replace_non_zero() -> DocReplaceFuncB:
    def replace_non_zero(text: SegmentedText, scores: List[int], word: List[int]) -> SegmentedText:
        assert_float_or_int(scores[0])
        drop_indices = [idx for idx, s in enumerate(scores) if s == 1]
        return get_replaced_segment(text, drop_indices, word)
    return replace_non_zero


def get_replace_zero() -> DocReplaceFuncB:
    def replace_zero(text: SegmentedText, scores: List[int], word: List[int]) -> SegmentedText:
        assert_float_or_int(scores[0])
        drop_indices = [idx for idx, s in enumerate(scores) if s == 0]
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


get_replace_not_related: Callable[[], DocReplaceFuncB] = get_replace_zero
get_replace_related: Callable[[], DocReplaceFuncB] = get_replace_non_zero


# Test TP vs FP (Precision)
# align(qt, dt)
# d = dt + w
# q = qt + w
PairReplaceFunc = Callable[[RelatedBinaryAnswer, RelatedEvalInstance, List[int]], SegmentedInstance]


def get_not_related_pair_replace_fn(target_idx) -> PairReplaceFunc:
    doc_modify_fn: DocReplaceFuncB = get_replace_not_related()
    query_modify_fn: QueryReplaceFunc = get_replace_non_target_query()
    pair_modify_fn = get_pair_modify_fn(query_modify_fn, doc_modify_fn, target_idx)
    return pair_modify_fn


# Test TN vs FN (Recall)
# align(qt, dt)
# d = d - dt + w
# q = qt + w
def get_related_pair_replace_fn(target_idx) -> PairReplaceFunc:
    doc_modify_fn: DocReplaceFuncB = get_replace_related()
    query_modify_fn: QueryReplaceFunc = get_replace_non_target_query()
    pair_modify_fn = get_pair_modify_fn(query_modify_fn, doc_modify_fn, target_idx)
    return pair_modify_fn


def get_pair_modify_fn(query_modify_fn: QueryReplaceFunc,
                       doc_modify_fn: DocReplaceFuncB,
                       target_seg_idx) -> PairReplaceFunc:
    def pair_modify_fn(answer: RelatedBinaryAnswer,
                       problem: RelatedEvalInstance,
                       word: List[int]) -> SegmentedInstance:

        doc_term_scores: List[int] = answer.score_table[target_seg_idx]
        new_doc: SegmentedText = doc_modify_fn(problem.seg_instance.text2,
                                               doc_term_scores, word)
        full_query = problem.seg_instance.text1
        new_query: SegmentedText = query_modify_fn(full_query, target_seg_idx, word)
        seg = SegmentedInstance(new_query, new_doc)
        return seg
    return pair_modify_fn


def get_partial_text_as_segment(text1: SegmentedText, target_seg_idx: int) -> SegmentedText:
    tokens = text1.get_tokens_for_seg(target_seg_idx)
    return SegmentedText.from_tokens_ids(tokens)


def ceil(f):
    eps = 1e-8
    return int(f - eps) + 1


class TenStepRandomDropPolicy(DropSamplePolicyIF):
    def __init__(self, discretize=False):
        self.discretize = discretize

    def get_drop_docs(self, text: SegmentedText, score_list: List[int]) -> List[SegmentedText]:
        n_pos = sum(score_list)
        segment_list = []
        for i in range(10):
            drop_rate = 1 - i / 10
            n_drop = ceil(n_pos * drop_rate)
            true_indices = [i for i in range(len(score_list)) if score_list[i]]
            n_drop = min(len(true_indices), n_drop)
            drop_indices = random.sample(true_indices, n_drop)
            if drop_indices:
                segment = text.get_dropped_text(drop_indices)
                segment_list.append(segment)
        return segment_list

    def combine_results(self, outcome_list: List[float]):
        if self.discretize:
            outcome_b = [1 if s >= 0.5 else 0 for s in outcome_list]
            return average(outcome_b)
        else:
            return average(outcome_list)


class TenStepRandomReplacePolicy(ReplaceSamplePolicyIF):
    def __init__(self, discretize=False):
        self.discretize = discretize

    def get_replaced_docs(self, text: SegmentedText, score_list: List[int], word: List[int]) -> List[SegmentedText]:
        n_pos = sum(score_list)
        segment_list = []
        for i in range(10):
            drop_rate = 1 - i / 10
            n_drop = int(n_pos * drop_rate)
            true_indices = [i for i in range(len(score_list)) if score_list[i]]
            n_drop = min(len(true_indices), n_drop)
            drop_indices = random.sample(true_indices, n_drop)
            if drop_indices:
                segment = get_replaced_segment(text, drop_indices, word)
                segment_list.append(segment)
        return segment_list

    def combine_results(self, outcome_list: List[float]):
        if self.discretize:
            outcome_b = [1 if s >= 0.5 else 0 for s in outcome_list]
            return average(outcome_b)
        else:
            return average(outcome_list)
