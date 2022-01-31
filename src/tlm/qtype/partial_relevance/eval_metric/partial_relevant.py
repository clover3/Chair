from typing import Tuple, List, NamedTuple

from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import get_max_idx
from tlm.qtype.partial_relevance.complement_search_pckg.complement_header import ComplementSearchOutput, PartialSegment
from tlm.qtype.partial_relevance.eval_data_structure import RelatedEvalInstance, RelatedEvalAnswer, SegmentedInstance
from tlm.qtype.partial_relevance.eval_metric.ep_common import TupleOfListFuture, EvalMetricWCIF
from tlm.qtype.partial_relevance.eval_metric.segment_modify_fn import DocModFunc
from tlm.qtype.partial_relevance.segmented_text import SegmentedText
from trainer.promise import PromiseKeeper, MyFuture, MyPromise, list_future


class EvalMetricPartialRelevant(EvalMetricWCIF):
    def __init__(self, forward_fn, seg_join_policy, preserve_seg_idx, doc_modify_fn: DocModFunc):
        self.seg_join_policy = seg_join_policy
        self.pk = PromiseKeeper(forward_fn, 0.035)
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
        else:
            eval_score = None
        return eval_score

    def do_duty(self):
        self.pk.do_duty(log_size=True)

#
#        1) For q, d where f(q, d) > 0.5
#        2) For dt_i,
#               a. score1(dt_i) = f(q, d) - f(q, d_i)
#        3) For c_i that are complement of ct,
#                                          score2(dt_i) = f(q' = (c_i, ct), d) - f(q' = (c_i, ct), d)
#        4) score(dt_i) = score1(dt_i) - score2(dt_i)
#


class FuturePerCase(NamedTuple):
    before_w_target_queries: List[MyFuture]
    before_wo_target_less_queries: List[MyFuture]
    after_w_target_queries: List[MyFuture]
    after_wo_target_queries: List[MyFuture]


class EvalMetricPartialRelevant2(EvalMetricWCIF):
    def __init__(self, forward_fn, seg_join_policy, target_seg_idx, doc_modify_fn: DocModFunc):
        self.seg_join_policy = seg_join_policy
        self.pk = PromiseKeeper(forward_fn, 0.035)
        self.target_seg_idx = target_seg_idx
        self.tokenizer = get_tokenizer()
        self.doc_modify_fn: DocModFunc = doc_modify_fn

    def combine_fn(self, text: SegmentedText, complement: PartialSegment) -> SegmentedText:
        drop_seg_idx = 1 - self.target_seg_idx
        return self.seg_join_policy.join_tokens(text, complement, drop_seg_idx)

    def seg_to_future(self, seg: SegmentedInstance) -> MyFuture:
        return MyPromise(seg, self.pk).future()

    def get_predictions_for_case(self,
                                 problem: RelatedEvalInstance,
                                 answer: RelatedEvalAnswer,
                                 complement: ComplementSearchOutput) -> FuturePerCase:
        before_target_less_queries = []
        for c in complement.complement_list:
            new_text1: SegmentedText = self.combine_fn(problem.seg_instance.text1, c)
            new_query_rel_doc = SegmentedInstance(new_text1, problem.seg_instance.text2)
            before_target_less_queries.append(self.seg_to_future(new_query_rel_doc))

        before_w_target_queries = [self.seg_to_future(problem.seg_instance)]

        new_text2 = self.doc_modify_fn(problem.seg_instance.text2,
                                       answer.contribution.table[self.target_seg_idx],
                                       )

        after_w_target_queries = [self.seg_to_future(SegmentedInstance(problem.seg_instance.text1, new_text2))]
        after_alt_query = []
        for c in complement.complement_list:
            new_text1 = self.combine_fn(problem.seg_instance.text1, c)
            new_query_rel_doc = SegmentedInstance(new_text1, new_text2)
            after_alt_query.append(self.seg_to_future(new_query_rel_doc))

        return FuturePerCase(before_w_target_queries,
                             before_target_less_queries,
                             after_w_target_queries,
                             after_alt_query)

    def convert_future_to_score(self, fpc: FuturePerCase) -> float:
        try:
            rel_w_target_before: float = max(list_future(fpc.before_w_target_queries))
            rel_w_target_after: float = max(list_future(fpc.after_w_target_queries))
            importance_w_target = rel_w_target_before - rel_w_target_after

            rel_wo_target_before: float = max(list_future(fpc.before_wo_target_less_queries))
            rel_wo_target_after: float = max(list_future(fpc.after_wo_target_queries))
            print(rel_w_target_before, rel_w_target_after, rel_wo_target_before, rel_wo_target_after)
            importance_wo_target = rel_wo_target_before - rel_wo_target_after
            eval_score = importance_w_target - importance_wo_target
        except ValueError:
            eval_score = None
        return eval_score

    def do_duty(self):
        self.pk.do_duty(log_size=True)
