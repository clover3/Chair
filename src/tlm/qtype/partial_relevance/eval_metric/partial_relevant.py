from typing import Tuple, List

import numpy as np

from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import get_max_idx, lmap
from misc_lib import two_digit_float
from tlm.qtype.partial_relevance.complement_search_pckg.complement_header import ComplementSearchOutput, PartialSegment
from tlm.qtype.partial_relevance.eval_metric.doc_modify_fns import DocModFunc
from tlm.qtype.partial_relevance.eval_data_structure import RelatedEvalInstance, RelatedEvalAnswer, SegmentedText, \
    SegmentedInstance, print_segment_text
from tlm.qtype.partial_relevance.eval_metric.ep_common import TupleOfListFuture, EvalMetricIF
from trainer.promise import PromiseKeeper, MyFuture, MyPromise, list_future


class EvalMetricPartialRelevant(EvalMetricIF):
    def __init__(self, forward_fn, seg_join_policy, preserve_seg_idx, doc_modify_fn: DocModFunc):
        self.seg_join_policy = seg_join_policy
        self.pk = PromiseKeeper(forward_fn)
        self.preserve_seg_idx = preserve_seg_idx
        self.tokenizer = get_tokenizer()
        self.doc_modify_fn: DocModFunc = doc_modify_fn

    def combine_fn(self, text: SegmentedText, complement: PartialSegment) -> SegmentedText:
        return self.seg_join_policy.join_tokens(text, complement, self.preserve_seg_idx)

    def seg_to_future(self, seg: SegmentedInstance) -> MyFuture:
        return MyPromise(seg, self.pk).future()

    def get_predictions_for_case(self,
                                 problem: RelatedEvalInstance,
                                 answer: RelatedEvalAnswer,
                                 complement: ComplementSearchOutput) -> TupleOfListFuture:
        before_futures = []
        for c in complement.complement_list:
            new_text1: SegmentedText = self.combine_fn(problem.seg_instance.text1, c)
            new_query_rel_doc = SegmentedInstance(new_text1, problem.seg_instance.text2)
            before_futures.append(self.seg_to_future(new_query_rel_doc))

        new_text2 = self.doc_modify_fn(problem.seg_instance.text2,
                                       answer.contribution.table[self.preserve_seg_idx],
                                       )

        if not complement.complement_list:
            print("No complement for ", problem.problem_id)
        else:
            print("Problem: {0}  initial_score={1:.2f}".format(problem.problem_id, problem.score))
            print("Before")
            print_segment_text(self.tokenizer, problem.seg_instance.text2)
            print("After")
            print_segment_text(self.tokenizer, new_text2)

        after_futures = []
        for c in complement.complement_list:
            new_text1 = self.combine_fn(problem.seg_instance.text1, c)
            new_query_rel_doc = SegmentedInstance(new_text1, new_text2)
            after_futures.append(self.seg_to_future(new_query_rel_doc))
        future_predictions = before_futures, after_futures
        return future_predictions

    def convert_future_to_score(self, future_prediction_list) -> float:
        def combine_to_partial_relevant(future_list):
            return max([f.get() for f in future_list])

        before, after = future_prediction_list
        if before and after:
            partial_relevant_before: float = combine_to_partial_relevant(before)
            partial_relevant_after: float = combine_to_partial_relevant(after)
            eval_score = partial_relevant_before - partial_relevant_after
            print("{0:.2f} -> {1:.2f}".format(partial_relevant_before, partial_relevant_after))
        else:
            eval_score = None
        return eval_score

    def print_future_score(self, future_prediction_list,
                           a_p_c: Tuple[RelatedEvalAnswer, RelatedEvalInstance, ComplementSearchOutput]):
        a, p, c = a_p_c
        def combine_to_partial_relevant(future_list):
            scores = list_future(future_list)
            i = get_max_idx(scores)
            s_complement = c.complement_list[i].to_text(self.tokenizer)
            print("Max complement: {0} ({1:2f}".format(s_complement, scores[i]))
            return max(scores)

        before, after = future_prediction_list
        if before and after:
            partial_relevant_before: float = combine_to_partial_relevant(before)
            partial_relevant_after: float = combine_to_partial_relevant(after)
            eval_score = partial_relevant_before - partial_relevant_after
            print("{0:.2f} -> {1:.2f}".format(partial_relevant_before, partial_relevant_after))
        else:
            eval_score = None
        return eval_score

    def do_duty(self):
        self.pk.do_duty(log_size=True)
