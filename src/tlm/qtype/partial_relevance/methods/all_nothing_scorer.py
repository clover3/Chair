import numpy as np

from bert_api.segmented_instance.seg_instance import SegmentedInstance
from contradiction.alignment.data_structure.eval_data_structure import ContributionSummary, MatrixScorerIF


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

