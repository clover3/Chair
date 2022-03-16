from bert_api.segmented_instance.seg_instance import SegmentedInstance
from tlm.qtype.partial_relevance.eval_data_structure import ContributionSummary, MatrixScorerIF


class ExactMatchScorer(MatrixScorerIF):
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

