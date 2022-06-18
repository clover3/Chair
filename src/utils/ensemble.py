from abc import ABC, abstractmethod
from typing import List, Callable

from list_lib import all_equal


def normalize(scores, cap=1.0):
    base = min(scores)
    gap = max(scores) - base
    if gap == 0:
        gap = 1

    def f(v):
        return (v - base) / gap * cap
    return [f(v) for v in scores]


class EnsembleCoreIF(ABC):
    @abstractmethod
    def combine(self, scores_list: List[List[float]]) -> List[float]:
        pass


ScoreTransformSig = Callable[[List[float]], List[float]]


class TransformEnsembleCore(EnsembleCoreIF):
    def __init__(self, score_transformer: List[ScoreTransformSig]):
        self.transform_list: List[ScoreTransformSig] = score_transformer

    def combine(self, scores_list: List[List[float]]) -> List[float]:
        if len(scores_list) == 0:
            return []
        new_score_list = [transform(scores) for scores, transform in zip(scores_list, self.transform_list)]
        if not all_equal(list(map(len, new_score_list))):
            raise ValueError

        maybe_len = len(scores_list[0])

        output = []
        for i in range(maybe_len):
            scores = [score_list[i] for score_list in new_score_list]
            s = sum(scores)
            output.append(s)
        return output


class NormalizeEnsembleCore(EnsembleCoreIF):
    def __init__(self, normalize_range: List[float]):
        self.cap_list = normalize_range
        score_transformer_l = [lambda scores: normalize(scores, cap) for cap in self.cap_list]
        self.core = TransformEnsembleCore(score_transformer_l)

    def combine(self, scores_list: List[List[float]]) -> List[float]:
        return self.core.combine(scores_list)


def get_even_ensembler(n_runs):
    norm_range = [1 for _ in range(n_runs)]
    return NormalizeEnsembleCore(norm_range)