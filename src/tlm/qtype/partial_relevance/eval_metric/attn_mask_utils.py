from typing import List, Tuple, Dict

from bert_api.segmented_instance.seg_instance import SegmentedInstance
from data_generator.tokenizer_wo_tf import JoinEncoder
from list_lib import left
from misc_lib import get_second
from tlm.qtype.partial_relevance.attention_based.attention_mask_eval import indices_to_mask_dict
from tlm.qtype.partial_relevance.attention_based.bert_masking_common import BERTMaskIF
from contradiction.alignment.data_structure.eval_data_structure import ContributionSummary


class BertMaskWrap:
    def __init__(self, client: BERTMaskIF, max_seq_length):
        self.client = client
        self.join_encoder = JoinEncoder(max_seq_length)

    def eval(self, items: List[Tuple[SegmentedInstance, Dict]]) -> List[List[float]]:
        payload = []
        for inst, mask_d in items:
            x0, x1, x2 = self.join_encoder.join(inst.text1.tokens_ids, inst.text2.tokens_ids)
            payload_item = x0, x1, x2, mask_d
            payload.append(payload_item)
        output = self.client.predict(payload)
        return output


def get_drop_mask(contrib: ContributionSummary,
                  drop_k: float,
                  inst: SegmentedInstance,
                  q_idx) -> Dict[Tuple[int, int], int]:
    l = []
    for d_idx in range(inst.text2.get_seg_len()):
        k = q_idx, d_idx
        v = contrib.table[q_idx][d_idx]
        l.append((k, v))
    l.sort(key=get_second, reverse=True)
    total_item = inst.text2.get_seg_len()
    keep_portion = 1 - drop_k
    n_item = int(total_item * keep_portion)
    drop_indices: List[Tuple[int, int]] = left(l[n_item:])
    mask_d: Dict[Tuple[int, int], int] = indices_to_mask_dict(drop_indices)
    return mask_d


def get_drop_mask_binary(binary_importance: List[List[int]],
                         inst: SegmentedInstance,
                         q_idx) -> Dict[Tuple[int, int], int]:
    drop_indices: List[Tuple[int, int]] = []
    for d_idx in range(inst.text2.get_seg_len()):
        k = q_idx, d_idx
        v = binary_importance[q_idx][d_idx]
        if v == 1:
            pass
        elif v == 0:
            drop_indices.append(k)
        else:
            assert False
    mask_d: Dict[Tuple[int, int], int] = indices_to_mask_dict(drop_indices)
    return mask_d




