import numpy as np

from tlm.qtype.partial_relevance.attention_based.attention_mask_eval import AttentionMaskScorerIF
from tlm.qtype.partial_relevance.eval_data_structure import SegmentedInstance, ContributionSummary


class RandomScorer(AttentionMaskScorerIF):
    def eval_contribution(self, inst: SegmentedInstance) -> ContributionSummary:
        l1 = inst.text1.get_seg_len()
        l2 = inst.text2.get_seg_len()
        scores = np.random.random([l1, l2])
        table = inst.score_np_table_to_table(scores)
        return ContributionSummary(table)

