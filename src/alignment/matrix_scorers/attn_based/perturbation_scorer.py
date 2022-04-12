from typing import Tuple, Dict, List

import numpy as np

from bert_api.segmented_instance.seg_instance import SegmentedInstance
from data_generator.tokenizer_wo_tf import JoinEncoder
from list_lib import lmap, left
from bert_api.bert_masking_common import BERTMaskIF
from alignment.data_structure.matrix_scorer_if import MatrixScorerIF, ContributionSummary


def apply_offset(new_mask: Dict[Tuple[int, int], int],
                 offset: int,
                 max_seq_length: int,
                 ) -> Dict[Tuple[int, int], int]:
    def convert_key(k):
        idx1, idx2 = k
        new_k = idx1, idx2 + offset
        return new_k

    d1 = {convert_key(k): v for k, v in new_mask.items()}
    return {k: v for k, v in d1.items() if k[1] < max_seq_length}


def mask_dict_assert(new_mask, max_seq_length):
    for k, v in new_mask.items():
        idx1, idx2 = k
        assert idx1 < max_seq_length
        assert idx2 < max_seq_length


class PerturbationScorer(MatrixScorerIF):
    def __init__(self, client: BERTMaskIF, max_seq_length, dist_fn):
        self.client = client
        self.max_seq_length = max_seq_length
        self.join_encoder = JoinEncoder(max_seq_length)
        self.dist_fn = dist_fn

    def eval_contribution(self, inst: SegmentedInstance) -> ContributionSummary:
        core_payload: List[Tuple[Tuple, np.array]] = []
        for q_seg_idx, d_seg_idx in inst.enum_seg_indice_pairs():
            v: np.array = inst.get_drop_mask(q_seg_idx, d_seg_idx)
            key = q_seg_idx, d_seg_idx
            core_payload.append((key, v))

        def enrich(k_v: Tuple) -> Tuple:
            k, v = k_v
            drop_mask = v
            new_mask: Dict = inst.translate_mask(drop_mask)
            offset = 2 + len(inst.text1.tokens_ids)
            new_mask = apply_offset(new_mask, offset, self.max_seq_length)
            x0, x1, x2 = self.join_encoder.join(inst.text1.tokens_ids, inst.text2.tokens_ids)

            mask_dict_assert(new_mask, self.max_seq_length)
            new_payload = x0, x1, x2, new_mask
            return new_payload

        base_inst: Tuple[Tuple[int, int], np.array] = ((-1, -1), inst.get_empty_mask())
        core_payload = [base_inst] + core_payload
        full_payload = lmap(enrich, core_payload)
        try:
            output = self.client.predict(full_payload)
            for x0, x1, x2, mask in full_payload:
                mask_dict_assert(mask, self.max_seq_length)
        except Exception:
            raise

        keys = left(core_payload)
        outputs_d = dict(zip(keys, output))

        base_score = outputs_d[-1, -1]
        contrib_score_d = {k: self.dist_fn(base_score, outputs_d[k]) for k in outputs_d}
        table = inst.score_d_to_table(contrib_score_d)

        # print("PerturbationScorer::eval_contribution exit")
        return ContributionSummary(table)
