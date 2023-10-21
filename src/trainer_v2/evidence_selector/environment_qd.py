import numpy as np

from typing import List, Tuple
from data_generator.tokenizer_wo_tf import get_tokenizer
from misc_lib import tensor_to_list
from trainer_v2.evidence_selector.defs import RLStateTensor
from trainer_v2.evidence_selector.environment import ConcatMaskStrategyI, PEInfo, IDS
from trainer_v2.evidence_selector.evidence_candidates import get_st_ed
from trainer_v2.evidence_selector.evidence_scoring import cross_entropy, mean_absolute_error


class ConcatMaskStrategyQD(ConcatMaskStrategyI):
    def __init__(self):
        tokenizer = get_tokenizer()
        self.mask_id = tokenizer.wordpiece_tokenizer.vocab["[MASK]"]
        self.cls_id = tokenizer.wordpiece_tokenizer.vocab["[CLS]"]
        self.sep_id = tokenizer.wordpiece_tokenizer.vocab["[SEP]"]

    def apply_mask(self, input_ids, segment_ids, action):
        select_mask_from_action: List[int] = action
        select_mask_np = np.array(select_mask_from_action, np.bool)
        is_first = np.logical_and(np.equal(segment_ids, 0), np.not_equal(input_ids, 0))
        select_mask_np = np.logical_or(select_mask_np, is_first)
        for v in [self.cls_id, self.sep_id, 0]:
            select_mask_np = np.logical_or(select_mask_np, np.equal(input_ids, v))
        input_ids_np = np.array(input_ids, np.int)
        new_input_ids = input_ids_np * select_mask_np + (1 - select_mask_np) * self.mask_id
        return new_input_ids

    def get_deletable_evidence_mask(self, input_ids, segment_ids):
        # return mask that indicates evidence part. it excludes CLS or SEP
        is_second = np.equal(segment_ids, 1)
        deletable_evidence = is_second
        for v in [self.cls_id, self.sep_id]:
            is_not_special_mask = np.not_equal(input_ids, v)
            deletable_evidence = np.logical_and(deletable_evidence, is_not_special_mask)
        return deletable_evidence

    def get_masked_input(self, state, action) -> Tuple[IDS, IDS]:
        input_ids, segment_ids = state
        new_input_ids = self.apply_mask(input_ids, segment_ids, action)
        return tensor_to_list(new_input_ids), tensor_to_list(segment_ids)

    def get_query_like_segment_mask(self, input_ids, segment_ids):
        is_first = np.logical_and(np.equal(segment_ids, 0), np.not_equal(input_ids, 0))
        return is_first


def get_pe_for_qd_ce(base_pred, rep_pred, action, state):
    return get_pe_for_qd_inner(base_pred, rep_pred, action, state, cross_entropy, 0.05, 0.05)


def get_pe_for_qd_mae(base_pred, rep_pred, action, state):
    return get_pe_for_qd_inner(base_pred, rep_pred, action, state, mean_absolute_error, 0.05, 0.05)


def get_pe_for_qd_inner(
        base_pred, rep_pred, action, state, get_error_fn, tolerance,
        density_weight,
):
    def get_doc_len(state: RLStateTensor):
        seg2_start, seg2_end = get_st_ed(state.segment_ids_np)
        return seg2_end - seg2_start

    def get_valid_action(state, action):
        action_np = np.array(action)
        is_second_seg = np.logical_and(np.equal(state.segment_ids_np, 1),
                                       np.not_equal(state.input_ids_np, 0))
        valid_action = np.multiply(action_np, is_second_seg.astype(np.int))
        return valid_action

    n_p_tokens = get_doc_len(state)
    valid_action = get_valid_action(state, action)
    num_used = int(np.sum(valid_action).tolist())
    # num_used = 5
    # n_p_tokens = 10
    return PEInfo(
        base_pred, rep_pred, num_used, n_p_tokens, get_error_fn, tolerance, density_weight)