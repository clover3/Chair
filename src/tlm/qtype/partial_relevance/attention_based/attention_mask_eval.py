import functools
from typing import List, Iterable, Dict, Tuple

import numpy as np
import scipy.special
from scipy.special import softmax

from data_generator.tokenizer_wo_tf import JoinEncoder
from list_lib import left
from misc_lib import get_second, two_digit_float
from tab_print import tab_print
from tlm.qtype.partial_relevance.attention_based.bert_masking_common import BERTMaskIF
from tlm.qtype.partial_relevance.eval_data_structure import ContributionSummary, SegmentedInstance
from tlm.qtype.partial_relevance.qd_segmented_instance import QDSegmentedInstance


def softmax_rev_sigmoid(logits):
    probs = scipy.special.softmax(logits, axis=-1)
    probs = probs[:, 1]
    return -np.log(1 / probs - 1)


def indices_to_mask_dict(indices: List[Tuple[int, int]]) -> Dict[Tuple[int, int], int]:
    d = {k: 1 for k in indices}
    return d


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


Logits = List[float]


def get_error(p1, p2):
    diff = np.array(p1) - np.array(p2)
    return np.linalg.norm(diff)


def get_drop_indices(l, keep_portion) -> List[Tuple[int, int]]:
    total_item = len(l)
    n_item = int(total_item * keep_portion)
    drop_indices: List[Tuple[int, int]] = left(l[n_item:])
    return drop_indices


class EvalPerQSeg:
    def __init__(self, client: BERTMaskIF, max_seq_length, dist_metric):
        self.client = client
        self.max_seq_length = max_seq_length
        self.join_encoder = JoinEncoder(max_seq_length)
        self.dist_metric = dist_metric

    def eval(self, inst: SegmentedInstance, contrib: ContributionSummary) -> List[float]:
        num_step = 10
        keep_portion_list = [i / num_step for i in range(num_step)]
        items = self.eval_inner(inst, contrib, keep_portion_list)

        def get_auc(pair: Tuple[Logits, List[Logits]]) -> float:
            base_point, points = pair
            l2_error = [self.dist_metric(base_point, p) for p in points]
            auc = sum(l2_error) / num_step
            return auc

        return list(map(get_auc, items))

    def eval_inner(self,
                   inst: SegmentedInstance,
                   contrib: ContributionSummary,
                   keep_portion_list: List[float]
                   ) -> List[Tuple[Logits, List[Logits]]]:
        x0, x1, x2 = self.join_encoder.join(inst.text1.tokens_ids, inst.text2.tokens_ids)
        items = []
        for q_idx in range(inst.text1.get_seg_len()):
            l = []
            for d_idx in range(inst.text2.get_seg_len()):
                k = q_idx, d_idx
                v = contrib.table[q_idx][d_idx]
                l.append((k, v))
            l.sort(key=get_second, reverse=True)
            total_item = inst.text2.get_seg_len()
            get_drop_indices_fn = functools.partial(get_drop_indices, l, total_item)
            drop_indices_per_case: Iterable[List[Tuple[int, int]]] = map(get_drop_indices_fn, keep_portion_list)
            mask_d_itr = map(indices_to_mask_dict, drop_indices_per_case)
            mask_d_itr_conv = map(inst.translate_mask, mask_d_itr)
            payload = []
            for mask_d in mask_d_itr_conv:
                payload_item = x0, x1, x2, mask_d
                payload.append(payload_item)

            base_item = x0, x1, x2, {}
            payload_ex = [base_item] + payload
            output = self.client.predict(payload_ex)
            points = output[1:]
            base_point = output[0]
            items.append((base_point, points))
        return items

    def verbose_print(self, inst: SegmentedInstance, contrib: ContributionSummary):
        x0, x1, x2 = self.join_encoder.join(inst.text1.tokens_ids, inst.text2.tokens_ids)
        for q_idx in range(inst.text1.get_seg_len()):
            l = []
            for d_idx in range(inst.text2.get_seg_len()):
                k = q_idx, d_idx
                v = contrib.table[q_idx][d_idx]
                l.append((k, v))
            l.sort(key=get_second, reverse=True)
            total_item = inst.text2.get_seg_len()
            payload = []
            num_step = 10

            keep_portion_list = [i / num_step for i in range(num_step)]
            n_item_list = [int(total_item * keep_portion) for keep_portion in keep_portion_list]

            for n_item in n_item_list:
                drop_indices: List[Tuple[int, int]] = left(l[n_item:])
                mask_d = indices_to_mask_dict(drop_indices)
                payload_item = x0, x1, x2, mask_d
                payload.append(payload_item)

            base_item = x0, x1, x2, {}
            payload_ex = [base_item] + payload
            output = self.client.predict(payload_ex)
            print(output)
            points = output[1:]
            base_point = output[0]
            def get_prob(logits):
                return softmax(logits, axis=-1)[1]

            print("Base prob", get_prob(base_point))
            prob_list = list(map(get_prob, points))
            tab_print("prob", "keep_portion", "n_item")
            for prob, keep_portion, n_item in zip(prob_list, keep_portion_list, n_item_list):
                tab_print(two_digit_float(prob), keep_portion, n_item)


