from typing import Tuple, Dict

from data_generator.tokenizer_wo_tf import JoinEncoder
from list_lib import lmap, left
from tlm.qtype.partial_relevance.attention_based.attention_mask_eval import AttentionMaskScorerIF
from tlm.qtype.partial_relevance.attention_based.bert_masking_common import BERTMaskIF
from tlm.qtype.partial_relevance.eval_data_structure import QDSegmentedInstance, ContributionSummary


class PerturbationScorer(AttentionMaskScorerIF):
    def __init__(self, client: BERTMaskIF, max_seq_length, dist_fn):
        self.client = client
        self.max_seq_length = max_seq_length
        self.join_encoder = JoinEncoder(max_seq_length)
        self.dist_fn = dist_fn

    def eval_contribution(self, inst: QDSegmentedInstance) -> ContributionSummary:
        # print("PerturbationScorer::eval_contribution ENTRY")
        core_payload = []
        for q_seg_idx, d_seg_idx in inst.enum_seg_indice_pairs():
            v = inst.get_drop_mask(q_seg_idx, d_seg_idx)
            key = q_seg_idx, d_seg_idx
            core_payload.append((key, v))

        def enrich(k_v) -> Tuple:
            k, v = k_v
            drop_mask = v
            new_mask: Dict = inst.translate_mask(drop_mask)
            x0, x1, x2 = self.join_encoder.join(inst.text1_tokens_ids, inst.text2_tokens_ids)
            new_payload = x0, x1, x2, new_mask
            return new_payload

        base_inst = (-1, -1), inst.get_empty_mask()
        core_payload = [base_inst] + core_payload
        full_payload = lmap(enrich, core_payload)
        # print("PerturbationScorer::eval_contribution predict {} items".format(len(full_payload)))
        output = self.client.predict(full_payload)

        keys = left(core_payload)
        outputs_d = dict(zip(keys, output))

        base_score = outputs_d[-1, -1]
        contrib_score_d = {k: self.dist_fn(base_score, outputs_d[k]) for k in outputs_d}
        table = inst.score_d_to_table(contrib_score_d)

        # print("PerturbationScorer::eval_contribution exit")
        return ContributionSummary(table)
