from typing import Tuple, List

import numpy as np

from data_generator.tokenizer_wo_tf import get_tokenizer
from misc_lib import tensor_to_list
from trainer_v2.chair_logging import c_log
from trainer_v2.evidence_selector.defs import RLStateTensor
from trainer_v2.evidence_selector.environment import PEInfoFromCount, IDS, concat_two_items, unconcat, \
    ConcatMaskStrategyI
from trainer_v2.evidence_selector.evidence_candidates import get_st_ed
from trainer_v2.reinforce.monte_carlo_policy_function import SA
from utils.xml_rpc_helper import ServerProxyEx


class ConcatMaskStrategyNLI(ConcatMaskStrategyI):
    def __init__(self):
        tokenizer = get_tokenizer()
        self.mask_id = tokenizer.wordpiece_tokenizer.vocab["[MASK]"]
        self.cls_id = tokenizer.wordpiece_tokenizer.vocab["[CLS]"]
        self.sep_id = tokenizer.wordpiece_tokenizer.vocab["[SEP]"]

    def apply_mask(self, input_ids, segment_ids, action):
        select_mask_for_p_tokens: List[int] = action
        select_mask_np = np.array(select_mask_for_p_tokens, np.bool)
        is_second = np.array(segment_ids, np.bool)
        select_mask_np = np.logical_or(select_mask_np, is_second)

        # always include these special tokens
        for v in [self.cls_id, self.sep_id, 0]:
            select_mask_np = np.logical_or(select_mask_np, np.equal(input_ids, v))

        input_ids_np = np.array(input_ids, np.int)
        new_input_ids = input_ids_np * select_mask_np + (1 - select_mask_np) * self.mask_id
        return new_input_ids

    def get_masked_input(self, state, action) -> Tuple[IDS, IDS]:
        input_ids, segment_ids = state
        new_input_ids = self.apply_mask(input_ids, segment_ids, action)
        return tensor_to_list(new_input_ids), tensor_to_list(segment_ids)


class PEPClientNLI:
    def __init__(self, server_addr, port):
        tokenizer = get_tokenizer()
        self.mask_id = tokenizer.wordpiece_tokenizer.vocab["[MASK]"]
        self.cls_id = tokenizer.wordpiece_tokenizer.vocab["[CLS]"]
        self.sep_id = tokenizer.wordpiece_tokenizer.vocab["[SEP]"]
        self.proxy = ServerProxyEx(server_addr, port)

    def get_masked_input(self, state_action) -> Tuple[IDS, IDS]:
        state, action = state_action
        input_ids, segment_ids = state
        select_mask_for_p_tokens: List[int] = action
        select_mask_np = np.array(select_mask_for_p_tokens, np.bool)
        is_second = np.array(segment_ids, np.bool)
        select_mask_np = np.logical_or(select_mask_np, is_second)

        # always include these special tokens
        for v in [self.cls_id, self.sep_id, 0]:
            select_mask_np = np.logical_or(select_mask_np, np.equal(input_ids, v))

        input_ids_np = np.array(input_ids, np.int)
        new_input_ids = input_ids_np * select_mask_np + (1 - select_mask_np) * self.mask_id
        return tensor_to_list(new_input_ids), tensor_to_list(segment_ids)

    def request(self, items: List[SA]) -> List[List[float]]:
        payload: List[Tuple[IDS, IDS]] = list(map(self.get_masked_input, items))
        concat_payload = concat_two_items(payload)
        c_log.debug("PEPClient send ENTRY")
        response = self.proxy.send(concat_payload)
        c_log.debug("PEPClient send DONE")
        output = unconcat(response, len(payload))
        return output


def get_pe_info_nli(base_pred, rep_pred, action, state):
    def get_n_p_tokens(state: RLStateTensor):
        seg2_start, _ = get_st_ed(state.segment_ids)
        h_st = 1
        h_ed = seg2_start - 1
        return h_ed - h_st

    def get_valid_action(state, action):
        action_np = np.array(action)
        is_first_seg = np.logical_and(np.equal(state.segment_ids, 0),
                                      np.not_equal(state.input_ids, 0))
        valid_action = np.multiply(action_np, is_first_seg.astype(np.int))
        return valid_action

    n_p_tokens = get_n_p_tokens(state)
    valid_action = get_valid_action(state, action)
    num_used = int(np.sum(valid_action).tolist())
    return PEInfoFromCount(base_pred, rep_pred, num_used, n_p_tokens)