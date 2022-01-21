import abc
from typing import Tuple, List

from tlm.qtype.partial_relevance.complement_search_pckg.complement_header import ComplementSearchOutput
from tlm.qtype.partial_relevance.eval_data_structure import RelatedEvalInstance, RelatedEvalAnswer
from trainer.promise import MyFuture

TupleOfListFuture = Tuple[List[MyFuture], List[MyFuture]]


class EvalMetricIF(abc.ABC):
    @abc.abstractmethod
    def get_predictions_for_case(self,
                                 problem: RelatedEvalInstance,
                                 answer: RelatedEvalAnswer,
                                 complement: ComplementSearchOutput) -> TupleOfListFuture:
        pass

    @abc.abstractmethod
    def convert_future_to_score(self, future_prediction_list) -> float:
        pass

    @abc.abstractmethod
    def do_duty(self):
        pass