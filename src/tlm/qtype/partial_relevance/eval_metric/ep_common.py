import abc
from typing import Tuple, List, TypeVar

from tlm.qtype.partial_relevance.complement_search_pckg.complement_header import ComplementSearchOutput
from tlm.qtype.partial_relevance.eval_data_structure import RelatedEvalInstance, RelatedEvalAnswer, RelatedBinaryAnswer, \
    RelatedEvalInstanceEx
from tlm.qtype.partial_relevance.segmented_text import SegmentedText
from trainer.promise import MyFuture

TupleOfListFuture = Tuple[List[MyFuture], List[MyFuture]]

T = TypeVar('T')


class EvalMetricIF(abc.ABC):
    @abc.abstractmethod
    def get_predictions_for_case(self,
                                 problem: RelatedEvalInstance,
                                 answer: RelatedEvalAnswer,
                                 ) -> T:
        pass

    @abc.abstractmethod
    def convert_future_to_score(self, future_prediction_list: T) -> float:
        pass

    @abc.abstractmethod
    def do_duty(self):
        pass


class EvalMetricBinaryIF(abc.ABC):
    @abc.abstractmethod
    def get_predictions_for_case(self,
                                 problem: RelatedEvalInstance,
                                 answer: RelatedBinaryAnswer,
                                 ) -> T:
        pass

    @abc.abstractmethod
    def convert_future_to_score(self, future_prediction_list: T) -> float:
        pass

    @abc.abstractmethod
    def do_duty(self):
        pass


class EvalMetricConditionalIF(abc.ABC):
    @abc.abstractmethod
    def get_condition_pf(self,
                                   problem: RelatedEvalInstance,
                                   answer: RelatedBinaryAnswer,
                                   ) -> T:
        pass

    @abc.abstractmethod
    def get_test_pf(self,
                            problem: RelatedEvalInstance,
                            answer: RelatedBinaryAnswer,
                            ) -> T:
        pass

    @abc.abstractmethod
    def convert_condition_pf(self, future):
        pass

    @abc.abstractmethod
    def convert_test_pf(self, future_prediction_list: T) -> float:
        pass

    @abc.abstractmethod
    def do_duty(self):
        pass


class EvalMetricConditionalPerTargetIF(abc.ABC):
    @abc.abstractmethod
    def get_condition_pf(self,
                         problem: RelatedEvalInstanceEx,
                         answer: RelatedBinaryAnswer,
                         ) -> T:
        pass

    @abc.abstractmethod
    def get_test_pf(self,
                    problem: RelatedEvalInstanceEx,
                    answer: RelatedBinaryAnswer,
                    ) -> T:
        pass

    @abc.abstractmethod
    def convert_condition_pf(self, future):
        pass

    @abc.abstractmethod
    def convert_test_pf(self, future_prediction_list: T) -> float:
        pass

    @abc.abstractmethod
    def do_duty(self):
        pass


Step1Output = TypeVar('Step1Output')
Step2Output = TypeVar('Step2Output')


class DropSamplePolicyIF(abc.ABC):
    @abc.abstractmethod
    def get_drop_docs(self,
                          text: SegmentedText,
                          score_list: List[int],
                          ) -> List[SegmentedText]:
        pass

    @abc.abstractmethod
    def combine_results(self, outcome_list: List[float]):
        pass


class ReplaceSamplePolicyIF(abc.ABC):
    @abc.abstractmethod
    def get_replaced_docs(self,
                          text: SegmentedText,
                          score_list: List[int],
                          word: List[int]) -> List[SegmentedText]:
        pass

    @abc.abstractmethod
    def combine_results(self, outcome_list: List[float]):
        pass


class EvalMetricWCIF(abc.ABC):
    @abc.abstractmethod
    def get_predictions_for_case(self,
                                 problem: RelatedEvalInstance,
                                 answer: RelatedEvalAnswer,
                                 complement: ComplementSearchOutput) -> T:
        pass

    @abc.abstractmethod
    def convert_future_to_score(self, future_prediction_list: T) -> float:
        pass

    @abc.abstractmethod
    def do_duty(self):
        pass


class EvalMetricBinaryWCIF(abc.ABC):
    @abc.abstractmethod
    def get_predictions_for_case(self,
                                 problem: RelatedEvalInstance,
                                 answer: RelatedBinaryAnswer,
                                 complement: ComplementSearchOutput) -> T:
        pass

    @abc.abstractmethod
    def convert_future_to_score(self, future_prediction_list: T) -> float:
        pass

    @abc.abstractmethod
    def do_duty(self):
        pass


class EvalMetricWCBuilder(EvalMetricBinaryWCIF):
    def __init__(self, inner: EvalMetricBinaryIF):
        self.inner = inner

    def get_predictions_for_case(self,
                                 problem: RelatedEvalInstance,
                                 answer: RelatedBinaryAnswer,
                                 complement: ComplementSearchOutput) -> TupleOfListFuture:
        return self.inner.get_predictions_for_case(problem, answer)

    def convert_future_to_score(self, future_prediction_list) -> float:
        return self.inner.convert_future_to_score(future_prediction_list)

    def do_duty(self):
        self.inner.do_duty()