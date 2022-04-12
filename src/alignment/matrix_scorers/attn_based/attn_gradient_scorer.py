import numpy as np

from bert_api.segmented_instance.seg_instance import SegmentedInstance
from alignment import MatrixScorerIF
from alignment.data_structure.matrix_scorer_if import ContributionSummary
from data_generator.tokenizer_wo_tf import JoinEncoder
from tlm.qtype.partial_relevance.attention_based.attention_mask_gradient import PredictorAttentionMaskGradient


class AttentionGradientScorer(MatrixScorerIF):
    def __init__(self, client: PredictorAttentionMaskGradient, max_seq_length):
        self.client = client
        self.max_seq_length = max_seq_length
        self.join_encoder = JoinEncoder(max_seq_length)

    def eval_contribution(self, inst: SegmentedInstance) -> ContributionSummary:
        x0, x1, x2 = self.join_encoder.join(inst.text1.tokens_ids, inst.text2.tokens_ids)
        payload_inst = x0, x1, x2, {}
        logits, grads = self.client.predict([payload_inst])
        grad_mag = np.sum(np.abs(grads), axis=3)
        assert len(grad_mag) == 1
        table = inst.score_np_table_to_table(grad_mag[0])
        return ContributionSummary(table)