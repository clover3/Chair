import numpy as np

from alignment.data_structure.matrix_scorer_if import MatrixScorerIF, ContributionSummary
from bert_api.segmented_instance.seg_instance import SegmentedInstance


class AllOneScorer(MatrixScorerIF):
    def eval_contribution(self, inst: SegmentedInstance) -> ContributionSummary:
        l1 = inst.text1.get_seg_len()
        l2 = inst.text2.get_seg_len()
        scores = np.ones([l1, l2])
        table = inst.score_np_table_to_table(scores)
        return ContributionSummary(table)


class AllZeroScorer(MatrixScorerIF):
    def eval_contribution(self, inst: SegmentedInstance) -> ContributionSummary:
        l1 = inst.text1.get_seg_len()
        l2 = inst.text2.get_seg_len()
        scores = np.zeros([l1, l2])
        table = inst.score_np_table_to_table(scores)
        return ContributionSummary(table)


