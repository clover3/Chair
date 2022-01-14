import abc
from typing import List, Tuple

import numpy as np
import scipy.special

from data_generator.tokenizer_wo_tf import JoinEncoder
from list_lib import left
from misc_lib import get_second
from tlm.qtype.partial_relevance.attention_based.bert_masking_common import BERTMaskIF
from tlm.qtype.partial_relevance.eval_data_structure import QDSegmentedInstance, ContributionSummary, SegmentedInstance


def softmax_rev_sigmoid(logits):
    probs = scipy.special.softmax(logits, axis=-1)
    probs = probs[:, 1]
    return -np.log(1 / probs - 1)


def indices_to_mask_dict(indices: List[Tuple[int, int]]):
    return {k: 1 for k in indices}


class EvalAll:
    def __init__(self, client: BERTMaskIF, max_seq_length):
        self.client = client
        self.max_seq_length = max_seq_length
        self.join_encoder = JoinEncoder(max_seq_length)

    def eval(self, inst: QDSegmentedInstance, contrib: ContributionSummary) -> float:
        l = []
        for q_idx, d_idx in inst.enum_seg_indice_pairs():
            k = q_idx, d_idx
            v = contrib.table[q_idx][d_idx]
            l.append((k, v))

        l.sort(key=get_second, reverse=True)
        x0, x1, x2 = self.join_encoder.join(inst.text1_tokens_ids, inst.text2_tokens_ids)

        total_item = inst.n_q_segs * inst.seg2_len
        payload_key = []
        payload = []
        num_step = 10
        for i in range(num_step):
            keep_portion = i / num_step
            n_item = int(total_item * keep_portion)
            drop_indices: List[Tuple[int, int]] = left(l[n_item:])
            mask_d = indices_to_mask_dict(drop_indices)
            payload_item = x0, x1, x2, mask_d
            payload.append(payload_item)
            payload_key.append(keep_portion)

        base_item = x0, x1, x2, {}
        payload_ex = [base_item] + payload
        output = self.client.predict(payload_ex)
        def get_error(p1, p2):
            diff = np.array(p1) - np.array(p2)
            return np.linalg.norm(diff)
        points = output[1:]
        base_point = output[0]
        l2_error = [get_error(base_point, p) for p in points]
        auc = sum(l2_error) / num_step
        return auc


class EvalPerQSeg:
    def __init__(self, client: BERTMaskIF, max_seq_length):
        self.client = client
        self.max_seq_length = max_seq_length
        self.join_encoder = JoinEncoder(max_seq_length)

    def eval(self, inst: QDSegmentedInstance, contrib: ContributionSummary) -> List[float]:
        x0, x1, x2 = self.join_encoder.join(inst.text1_tokens_ids, inst.text2_tokens_ids)
        auc_list = []
        for q_idx in range(inst.get_seg1_len()):
            l = []
            for d_idx in range(inst.get_seg2_len()):
                k = q_idx, d_idx
                v = contrib.table[q_idx][d_idx]
                l.append((k, v))
            l.sort(key=get_second, reverse=True)
            total_item = inst.get_seg2_len()
            payload = []
            num_step = 10
            for i in range(num_step):
                keep_portion = i / num_step
                n_item = int(total_item * keep_portion)
                drop_indices: List[Tuple[int, int]] = left(l[n_item:])
                mask_d = indices_to_mask_dict(drop_indices)
                payload_item = x0, x1, x2, mask_d
                payload.append(payload_item)

            base_item = x0, x1, x2, {}
            payload_ex = [base_item] + payload
            output = self.client.predict(payload_ex)
            def get_error(p1, p2):
                diff = np.array(p1) - np.array(p2)
                return np.linalg.norm(diff)
            points = output[1:]
            base_point = output[0]
            l2_error = [get_error(base_point, p) for p in points]
            auc = sum(l2_error) / num_step
            auc_list.append(auc)
        return auc_list


class AttentionMaskScorerIF(abc.ABC):
    @abc.abstractmethod
    def eval_contribution(self, inst: SegmentedInstance) -> ContributionSummary:
        pass