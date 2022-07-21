from typing import List

from alignment.data_structure.matrix_scorer_if import MatrixScorerIF, ContributionSummary, MatrixScorerIF2
from bert_api.segmented_instance.seg_instance import SegmentedInstance
from list_lib import list_equal


class TokenExactMatchScorer(MatrixScorerIF):
    def eval_contribution(self, inst: SegmentedInstance) -> ContributionSummary:
        l1 = inst.text1.get_seg_len()
        l2 = inst.text2.get_seg_len()

        output_table = []
        for i1 in range(l1):
            scores_per_seg = []
            tokens = inst.text1.get_tokens_for_seg(i1)
            for i2 in range(l2):
                score = 0
                for i2_i in inst.text2.get_tokens_for_seg(i2):
                    if i2_i in tokens:
                        score += 1
                scores_per_seg.append(score)
            output_table.append(scores_per_seg)

        return ContributionSummary(output_table)


class SegmentExactMatchScorer(MatrixScorerIF):
    def eval_contribution(self, inst: SegmentedInstance) -> ContributionSummary:
        l1 = inst.text1.get_seg_len()
        l2 = inst.text2.get_seg_len()

        output_table = []
        for i1 in range(l1):
            scores_per_seg: List[float] = []
            tokens1: List[int] = inst.text1.get_tokens_for_seg(i1)
            for i2 in range(l2):
                tokens2: List[int] = inst.text2.get_tokens_for_seg(i2)
                score = 1 if list_equal(tokens1, tokens2) else 0
                scores_per_seg.append(score)
            output_table.append(scores_per_seg)

        return ContributionSummary(output_table)


class ExactMatchScorer2(MatrixScorerIF2):
    def solve(self, tokens1: List[str], tokens2: List[str]) -> List[List[float]]:
        l1 = len(tokens1)
        l2 = len(tokens2)

        def normalize(s):
            return s.lower()

        output_table = []
        for i1 in range(l1):
            scores_per_seg: List[float] = []
            for i2 in range(l2):
                t1 = normalize(tokens1[i1])
                t2 = normalize(tokens2[i2])
                score = 1 if t1 == t2 else 0
                scores_per_seg.append(score)
            output_table.append(scores_per_seg)
        return output_table

