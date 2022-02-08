import abc
from typing import Tuple, List, TypeVar

from tlm.qtype.partial_relevance.complement_search_pckg.complement_header import ComplementSearchOutput
from tlm.qtype.partial_relevance.eval_data_structure import RelatedEvalInstance, RelatedEvalAnswer, RelatedBinaryAnswer
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