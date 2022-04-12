from typing import Tuple, List, Any

from bert_api.segmented_instance.seg_instance import SegmentedInstance
from bert_api.segmented_instance.segmented_text import SegmentedText
from data_generator.tokenizer_wo_tf import get_tokenizer
from tlm.qtype.partial_relevance.complement_search_pckg.complement_header import ComplementSearchOutput, PartialSegment
from alignment.data_structure.related_eval_instance import RelatedEvalInstance
from tlm.qtype.partial_relevance.eval_metric.ep_common import TupleOfListFuture
from tlm.qtype.partial_relevance.eval_metric.segment_modify_fn import DocModFuncR
from trainer.promise import PromiseKeeper, MyFuture, MyPromise

DMC = Any
FutureForCase = Tuple[List[MyFuture], List[Tuple[DMC, List[MyFuture]]]]


class SearchPartialRelevant:
    def __init__(self, forward_fn, seg_join_policy, doc_modify_fn, preserve_seg_idx):
        self.seg_join_policy = seg_join_policy
        self.pk = PromiseKeeper(forward_fn, 0.035)
        self.preserve_seg_idx = preserve_seg_idx
        self.tokenizer = get_tokenizer()
        self.doc_modify_fn: DocModFuncR = doc_modify_fn

    def combine_fn(self, text: SegmentedText, complement: PartialSegment) -> SegmentedText:
        return self.seg_join_policy.join_tokens(text, complement, self.preserve_seg_idx)

    def seg_to_future(self, seg: SegmentedInstance) -> MyFuture:
        return MyPromise(seg, self.pk).future()

    def get_predictions_for_case(self,
                                 problem: RelatedEvalInstance,
                                 complement: ComplementSearchOutput,
                                 doc_modify_candidate: DMC,
                                 ) -> TupleOfListFuture:
        before_futures: List[MyFuture] = []
        for c in complement.complement_list:
            new_text1: SegmentedText = self.combine_fn(problem.seg_instance.text1, c)
            new_query_rel_doc = SegmentedInstance(new_text1, problem.seg_instance.text2)
            before_futures.append(self.seg_to_future(new_query_rel_doc))

        new_text2 = self.doc_modify_fn(problem.seg_instance.text2, doc_modify_candidate)
        after_futures: List[MyFuture] = []
        for c in complement.complement_list:
            new_text1 = self.combine_fn(problem.seg_instance.text1, c)
            new_query_rel_doc = SegmentedInstance(new_text1, new_text2)
            after_futures.append(self.seg_to_future(new_query_rel_doc))
        future_predictions: TupleOfListFuture = before_futures, after_futures
        return future_predictions

    def convert_future_to_score(self, future_prediction_list: TupleOfListFuture) -> float:
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


    def do_duty(self):
        self.pk.do_duty(log_size=True)
