import abc
from abc import ABC, abstractmethod
from typing import List, TypeVar, Optional

from contradiction.alignment.data_structure.eval_data_structure import RelatedEvalInstanceEx, RelatedBinaryAnswer


class EvalMetricV3IF(abc.ABC):
    @abc.abstractmethod
    def get_first_state(self, p: RelatedEvalInstanceEx, a: RelatedBinaryAnswer):
        pass

    @abc.abstractmethod
    def apply_map(self, state_list: List):
        pass

    @abc.abstractmethod
    def apply_reduce(self, t_future_list, state_list: List):
        pass

    @abc.abstractmethod
    def do_duty(self):
        pass

    @abc.abstractmethod
    def get_scores(self, state_list):
        pass

    @abc.abstractmethod
    def is_final(self, code):
        pass


class EvalV3StateIF(ABC):
    @abstractmethod
    def get_code(self):
        pass


# Inherit MetricV3 to reuse functions: apply_map, apply_reduce
class MetricV3(EvalMetricV3IF):
    @abstractmethod
    def get_state_worker(self, state_code):
        pass

    @abstractmethod
    def get_first_state(self, p: RelatedEvalInstanceEx, a: RelatedBinaryAnswer):
        pass

    def apply_map(self, state_list: List[EvalV3StateIF]):
        t_future_list = []
        for state in state_list:
            worker: V3StateWorkerIF = self.get_state_worker(state.get_code())
            t_future = worker.map(state)
            t_future_list.append(t_future)
        return t_future_list

    def apply_reduce(self, t_future_list, state_list: List[EvalV3StateIF]):
        next_state_list: List[EvalV3StateIF] = []
        for t_future, state in zip(t_future_list, state_list):
            new_state = self.get_state_worker(state.get_code()).reduce(t_future, state)
            next_state_list.append(new_state)

        return next_state_list

    @abstractmethod
    def do_duty(self):
        pass

    def get_scores(self, state_list: List):
        return [s.score for s in state_list]


T = TypeVar('T')


class V3StateWorkerIF(ABC):
    @abstractmethod
    def map(self, item: EvalV3StateIF) -> T:
        # prepares the calculation
        # register required calculation to promise keeper
        # promise keeper
        pass

    @abstractmethod
    def reduce(self, obj: T, item: EvalV3StateIF) -> EvalV3StateIF:
        # Given a future value, return next State object
        pass


class UnexpectedCodeException(Exception):
    def __init__(self, code):
        self.code = code

    def __str__(self):
        return f"Code {self.code} is not expected"


def assert_code(actual_code, expected_code):
    if expected_code != actual_code:
        raise UnexpectedCodeException(actual_code)


class StateDone(EvalV3StateIF):
    def __init__(self,
                 problem: RelatedEvalInstanceEx,
                 answer: RelatedBinaryAnswer,
                 score: Optional[float]
                 ):
        self.problem = problem
        self.answer = answer
        self.score = score

    @abstractmethod
    def get_code(self):
        pass

    def get_score(self):
        return self.score


class FinalStateWorker(V3StateWorkerIF):
    def __init__(self, expected_code):
        self.expected_code = expected_code

    def map(self, item: StateDone) -> None:
        assert_code(item.get_code(), self.expected_code)
        return None

    def reduce(self, none: None,
               item: StateDone) -> StateDone:
        return item
