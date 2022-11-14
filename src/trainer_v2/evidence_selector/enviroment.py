import random
from typing import List, Tuple, Any, NamedTuple

from data_generator.tokenizer_wo_tf import get_tokenizer
from misc_lib import ceil_divide, tensor_to_list
from port_info import LOCAL_DECISION_PORT
from trainer_v2.chair_logging import c_log
from trainer_v2.evidence_selector.evidence_candidates import get_st_ed
from trainer_v2.evidence_selector.evidence_scoring import cross_entropy, length_loss
from trainer_v2.evidence_selector.defs import RLStateTensor
from trainer_v2.reinforce.monte_carlo_policy_function import SA
from utils.xml_rpc_helper import ServerProxyEx
import numpy as np


# PE: Partial Evidence
class PEInfo(NamedTuple):
    base_pred: List[float]
    rep_pred: List[float]
    num_used: int
    n_p_tokens: int

    def ce_error(self):
        err = cross_entropy(np.array(self.base_pred), np.array(self.rep_pred))  # [0, inf]
        return err

    def density(self):
        return length_loss(self.num_used, self.n_p_tokens)

    def combined_score(self):
        tolerance = 0.05
        err_cap = 10
        err = self.ce_error()
        err = min(err, err_cap)  # [0, 5]
        err = max(tolerance, err)
        combined_score = (err_cap - err) - tolerance * self.density()
        return combined_score


class PEPEnvironment:
    def __init__(self, server):
        self.server = server

    def request(self, items: List[Tuple[RLStateTensor, List[int]]]) -> List[List[float]]:
        def transform(sa):
            state, action = sa
            state_raw = state.input_ids, state.segment_ids
            return state_raw, action

        client = PEPClient(self.server, LOCAL_DECISION_PORT)
        ret = client.request(list(map(transform, items)))
        return ret

    def get_item_results(self, items: List[Tuple[RLStateTensor, List[int]]]) -> List[PEInfo]:
        # Build dictionary of base predictions
        base_items = {}
        for sa in items:
            state, action = sa
            key = state.input_ids_hash()
            if key not in base_items:
                no_del_action = [1] * len(state.input_ids)
                base_items[key] = state, no_del_action

        bases_to_calculate = list(base_items.values())
        payload = bases_to_calculate + items

        base_preds = {}
        outputs: List[List[float]] = self.request(payload)

        base_outputs = outputs[:len(bases_to_calculate)]
        item_outputs = outputs[len(bases_to_calculate):]

        assert len(item_outputs) == len(items)

        for sa, output in zip(bases_to_calculate, base_outputs):
            state, _ = sa
            base_preds[state.input_ids_hash()] = output

        def get_n_p_tokens(state: RLStateTensor):
            seg2_start, _ = get_st_ed(state.segment_ids)
            h_st = 1
            h_ed = seg2_start -1
            return h_ed - h_st

        def num_token_in_action(state, action) -> int:
            action_np = np.array(action)
            is_first_seg = np.logical_and(np.equal(state.segment_ids, 0),
                                         np.not_equal(state.input_ids, 0))
            valid_action = np.multiply(action_np, is_first_seg.astype(np.int))
            return int(np.sum(valid_action).tolist())

        d_n_p_tokens = dict()
        pe_result_list: List[PEInfo] = []
        for sa, output in zip(items, item_outputs):
            output: List[float] = output
            state, action = sa
            base_output: List[float] = base_preds[state.input_ids_hash()]
            try:
                n_p_tokens = d_n_p_tokens[state.input_ids_hash()]
            except KeyError:
                n_p_tokens = get_n_p_tokens(state)

            num_used = num_token_in_action(state, action)
            pe_result = PEInfo(base_output, output, num_used, n_p_tokens)
            pe_result_list.append(pe_result)

        assert len(pe_result_list) == len(items)
        return pe_result_list


IDS = List[int]
def concat_two_items(payload):
    def concat_item(item1, item2):
        input_ids1, segment_ids1 = item1
        input_ids2, segment_ids2 = item2
        return input_ids1 + input_ids2, segment_ids1 + segment_ids2

    concat_items = []
    n_item_d2 = ceil_divide(len(payload), 2)
    for i in range(n_item_d2):
        if i * 2 + 1 < len(payload):
            item = concat_item(payload[i * 2], payload[i * 2 + 1])
        else:
            item = concat_item(payload[i * 2], payload[i * 2])
        concat_items.append(item)
    return concat_items


def unconcat(output, n_original):
    l_decision_list: list[Any] = []
    for item in output:
        l_decision, g_decision = item
        l_decision1 = l_decision[0]
        l_decision2 = l_decision[1]
        l_decision_list.append(l_decision1)
        if len(l_decision_list) + 1 <= n_original:
            l_decision_list.append(l_decision2)
    return l_decision_list



def pretty_input_ids(t):
    l = tensor_to_list(t)
    return [t for t in l if t!=0]


# PEP: Partial Evidence Predictor
class PEPClient:
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
        select_mask_np = np.logical_or(select_mask_np, np.array(segment_ids, np.bool))
        for v in [self.cls_id, self.sep_id, 0]:
            select_mask_np = np.logical_or(select_mask_np, np.equal(input_ids, v))
        input_ids_np = np.array(input_ids, np.int)
        new_input_ids = input_ids_np * select_mask_np + (1-select_mask_np) * self.mask_id
        return tensor_to_list(new_input_ids), tensor_to_list(segment_ids)

    def request(self, items: List[SA]) -> List[List[float]]:
        payload: List[Tuple[IDS, IDS]] = list(map(self.get_masked_input, items))
        concat_payload = concat_two_items(payload)
        c_log.debug("PEPClient send ENTRY")
        response = self.proxy.send(concat_payload)
        c_log.debug("PEPClient send DONE")
        output = unconcat(response, len(payload))
        return output


def main():
    item = [0 for _ in range(300)]
    payload = [(item, (item, item))] * 5
    client = PEPClient("localhost", LOCAL_DECISION_PORT)
    output = client.request(payload)
    for e in output:
        print(e)


if __name__ == "__main__":
    main()


