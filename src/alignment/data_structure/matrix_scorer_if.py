import abc
from typing import NamedTuple, List

from alignment.data_structure.related_eval_instance import RelatedEvalInstance
from bert_api.segmented_instance.seg_instance import SegmentedInstance


class ContributionSummary(NamedTuple):
    table: List[List[float]]

    @classmethod
    def from_single_array(cls, arr: List[float], target_seg_idx, n_seg):
        output = []
        for i in range(n_seg):
            if i == target_seg_idx:
                output.append(arr)
            else:
                output.append([])
        return ContributionSummary(output)

    @classmethod
    def from_indices(cls, indices: List[int], target_seg_idx, p: RelatedEvalInstance):
        n = p.seg_instance.text2.get_seg_len()
        zeros = [0 for _ in range(n)]
        for i in indices:
            zeros[i] = 1

        return cls.from_single_array(zeros, target_seg_idx, p.seg_instance.text1.get_seg_len())


class MatrixScorerIF(abc.ABC):
    @abc.abstractmethod
    def eval_contribution(self, inst: SegmentedInstance) -> ContributionSummary:
        pass


class MatrixScorerIF2(abc.ABC):
    @abc.abstractmethod
    def solve(self, tokens1: List[str], tokens2: List[str]) -> List[List[float]]:
        pass

