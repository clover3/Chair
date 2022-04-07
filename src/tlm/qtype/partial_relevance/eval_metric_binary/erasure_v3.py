from typing import Callable, List

from bert_api.segmented_instance.seg_instance import SegmentedInstance
from bert_api.segmented_instance.segmented_text import SegmentedText
from data_generator.tokenizer_wo_tf import get_tokenizer
from misc_lib import threshold05
from contradiction.alignment.data_structure.eval_data_structure import RelatedEvalInstanceEx, \
    RelatedBinaryAnswer
from tlm.qtype.partial_relevance.eval_metric.ep_common import DropSamplePolicyIF
from tlm.qtype.partial_relevance.eval_metric.segment_modify_fn import get_partial_text_as_segment, get_drop_zero
from tlm.qtype.partial_relevance.eval_metric_binary.eval_v3_common import EvalV3StateIF
from tlm.qtype.partial_relevance.eval_metric_binary.one_step_common import SingleStateEvalMetric, SingleStateWorkerIF, \
    OneStepStateBegin, OneStepStateDone
from trainer.promise import PromiseKeeper, MyFuture, MyPromise, list_future


class ErasureWorker(SingleStateWorkerIF):
    def __init__(self,
                 forward_fn: Callable[[List[SegmentedInstance]], List[float]],
                 drop_sample_policy: DropSamplePolicyIF):
        self.drop_sample_policy = drop_sample_policy
        self.tokenizer = get_tokenizer()
        self.forward_fn: Callable[[List[SegmentedInstance]], List[float]] = forward_fn
        self.pk = PromiseKeeper(self.forward_fn, 0.035)

    def map(self, state: OneStepStateBegin) -> List[MyFuture[float]]:
        problem: RelatedEvalInstanceEx = state.problem
        answer: RelatedBinaryAnswer = state.answer
        new_query = get_partial_text_as_segment(problem.seg_instance.text1, problem.target_seg_idx)
        drop_doc_list: List[SegmentedText] = self.drop_sample_policy.get_drop_docs(
            problem.seg_instance.text2,
            answer.score_table[problem.target_seg_idx],
        )
        si_list: List[SegmentedInstance] = [SegmentedInstance(new_query, d) for d in drop_doc_list]
        future_predictions: List[MyFuture[float]] = [MyPromise(si, self.pk).future() for si in si_list]
        return future_predictions

    def reduce(self, future_prediction_list, state: OneStepStateBegin) -> EvalV3StateIF:
        score_list = list_future(future_prediction_list)
        score = self.drop_sample_policy.combine_results(score_list)
        return OneStepStateDone(state.problem, state.answer, score)

    def do_duty(self):
        self.pk.do_duty(log_size=True, reset=True)


class ErasureV3(SingleStateEvalMetric):
    def __init__(self, forward_fn, drop_sample_policy):
        worker = ErasureWorker(forward_fn, drop_sample_policy)
        super(ErasureV3, self).__init__(worker)


class ErasureSufficiencyWorker(SingleStateWorkerIF):
    def __init__(self,
                 forward_fn: Callable[[List[SegmentedInstance]], List[float]],
                 discretize):
        self.tokenizer = get_tokenizer()
        self.forward_fn: Callable[[List[SegmentedInstance]], List[float]] = forward_fn
        self.pk = PromiseKeeper(self.forward_fn, 0.035)
        self.drop_zero = get_drop_zero()
        self.discretize = discretize

    def map(self, state: OneStepStateBegin) -> MyFuture[float]:
        problem: RelatedEvalInstanceEx = state.problem
        answer: RelatedBinaryAnswer = state.answer
        new_query = get_partial_text_as_segment(problem.seg_instance.text1, problem.target_seg_idx)
        new_doc = self.drop_zero(problem.seg_instance.text2,
                                 answer.score_table[problem.target_seg_idx])
        si: SegmentedInstance = SegmentedInstance(new_query, new_doc)
        future_prediction: MyFuture[float] = MyPromise(si, self.pk).future()
        return future_prediction

    def reduce(self, future_prediction, state: OneStepStateBegin) -> EvalV3StateIF:
        score = future_prediction.get()
        if self.discretize:
            score = threshold05(score)
        return OneStepStateDone(state.problem, state.answer, score)

    def do_duty(self):
        self.pk.do_duty(log_size=True, reset=True)


class ErasureSufficiency(SingleStateEvalMetric):
    def __init__(self, forward_fn, discretize):
        worker = ErasureSufficiencyWorker(forward_fn, discretize)
        super(ErasureSufficiency, self).__init__(worker)

