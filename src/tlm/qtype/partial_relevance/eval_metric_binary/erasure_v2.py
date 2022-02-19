import functools
from typing import List, Tuple, Callable

from data_generator.tokenizer_wo_tf import get_tokenizer
from tlm.qtype.partial_relevance.eval_data_structure import RelatedEvalInstance, SegmentedInstance, RelatedBinaryAnswer, \
    RelatedEvalInstanceEx
from tlm.qtype.partial_relevance.eval_metric.ep_common import EvalMetricConditionalIF, DropSamplePolicyIF, \
    EvalMetricConditionalPerTargetIF
from tlm.qtype.partial_relevance.eval_metric.segment_modify_fn import get_partial_text_as_segment
from tlm.qtype.partial_relevance.segmented_text import SegmentedText
from trainer.promise import MyPromise, PromiseKeeper, MyFuture, list_future


class ErasureV2_single_seg(EvalMetricConditionalIF):
    def __init__(self, forward_fn, drop_sample_policy: DropSamplePolicyIF, target_seg_idx):
        self.pk = PromiseKeeper(forward_fn, 0.035)
        self.drop_sample_policy = drop_sample_policy
        self.target_seg_idx = target_seg_idx
        self.tokenizer = get_tokenizer()

    def seg_to_future(self, seg: SegmentedInstance) -> MyFuture:
        return MyPromise(seg, self.pk).future()

    def get_future(self, text1, text2):
        return self.seg_to_future(SegmentedInstance(text1, text2))

    def get_condition_pf(self,
                         problem: RelatedEvalInstance,
                         answer: RelatedBinaryAnswer,
                         ) -> MyFuture[float]:
        new_query = get_partial_text_as_segment(problem.seg_instance.text1, self.target_seg_idx)
        qd = self.get_future(new_query, problem.seg_instance.text2)
        return qd

    def get_test_pf(self,
                    problem: RelatedEvalInstance,
                    answer: RelatedBinaryAnswer,
                    ):
        new_query = get_partial_text_as_segment(problem.seg_instance.text1, self.target_seg_idx)
        drop_doc_list: List[SegmentedText] = self.drop_sample_policy.get_drop_docs(
            problem.seg_instance.text2,
            answer.score_table[self.target_seg_idx],
        )
        get_future_fn = functools.partial(self.get_future, new_query)
        future_predictions = list(map(get_future_fn, drop_doc_list))
        return future_predictions

    def convert_condition_pf(self, future_prediction: MyFuture[float]) -> bool:
        score = future_prediction.get()
        return score >= 0.5

    def convert_test_pf(self, future_prediction_list) -> float:
        score_list = list_future(future_prediction_list)
        return self.drop_sample_policy.combine_results(score_list)

    def do_duty(self):
        self.pk.do_duty(log_size=True)


class ErasureV2(EvalMetricConditionalPerTargetIF):
    def __init__(self,
                 forward_fn: Callable[[List[SegmentedInstance]], List[float]],
                 drop_sample_policy: DropSamplePolicyIF):
        self.drop_sample_policy = drop_sample_policy
        self.tokenizer = get_tokenizer()
        self.forward_fn: Callable[[List[SegmentedInstance]], List[float]] = forward_fn
        self.pk = PromiseKeeper(self.forward_fn, 0.035)

    def get_condition_pf(self,
                         problem: RelatedEvalInstanceEx,
                         answer: RelatedBinaryAnswer,
                         ) -> MyFuture[List[Tuple[int, float]]]:
        new_query = get_partial_text_as_segment(problem.seg_instance.text1, problem.target_seg_idx)
        si = SegmentedInstance(new_query, problem.seg_instance.text2)
        return MyPromise(si, self.pk).future()

    def get_test_pf(self,
                    problem: RelatedEvalInstanceEx,
                    answer: RelatedBinaryAnswer,
                    ) -> List[MyFuture[float]]:
        new_query = get_partial_text_as_segment(problem.seg_instance.text1, problem.target_seg_idx)
        drop_doc_list: List[SegmentedText] = self.drop_sample_policy.get_drop_docs(
            problem.seg_instance.text2,
            answer.score_table[problem.target_seg_idx],
        )
        si_list: List[SegmentedInstance] = [SegmentedInstance(new_query, d) for d in drop_doc_list]
        future_predictions: List[MyFuture[float]] = [MyPromise(si, self.pk).future() for si in si_list]
        return future_predictions

    def convert_condition_pf(self, future_prediction: MyFuture[float]) -> bool:
        score = future_prediction.get()
        return score >= 0.5

    def convert_test_pf(self, future_prediction_list) -> float:
        score_list = list_future(future_prediction_list)
        return self.drop_sample_policy.combine_results(score_list)

    def do_duty(self):
        self.pk.do_duty(log_size=True, reset=True)
