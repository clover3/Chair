import random
from typing import List

from typing import List, Iterable, Callable, Dict, Tuple, Set
import numpy as np
import tensorflow as tf

from cache import save_to_pickle
from models.transformer.bert_common_v2 import get_shape_list2
from trainer_v2.chair_logging import c_log
from trainer_v2.evidence_selector.defs import RLStateTensor
from trainer_v2.evidence_selector.evidence_candidates import get_st_ed
from trainer_v2.evidence_selector.seq_pred_policy_gradient import batch_rl_state_list
from trainer_v2.reinforce.monte_carlo_policy_function import PolicyFunction, Action
import scipy.special

# In this policy function, we only select one or two continuous segments as evidence
#
#


def set_value_by_range(total_len, st, ed, val_if_inside, val_if_outside) -> Action:
    action = []
    for i in range(total_len):
        if st <= i < ed:
            v = val_if_inside
        else:
            v = val_if_outside
        action.append(v)
    return np.array(action)


Action = List[int]
class SequenceLabelPolicyFunction(PolicyFunction):
    # State : input_ids, segment_ids
    def __init__(self, model, num_sample=10, target_segment="first"):
        self.model = model
        self.num_sample = num_sample
        self.epsilon = 0.5
        self.target_segment = target_segment

        self.SEP_ID = 102
        self.CLS_ID = 101

    def __dummy_model(self, item):
        input_ids, segment_ids = item
        f = tf.ones_like(input_ids, tf.float32)
        ret = tf.expand_dims(f, axis=2)
        return ret

    # enum all
    def enum_all_possible_action(self, state: RLStateTensor) -> List[Action]:
        segment_ids = state.segment_ids_np
        input_ids = state.input_ids_np
        # Decide size, enum all candidate actions
        first_segment = np.logical_and(np.equal(segment_ids, 0), np.not_equal(input_ids, 0))
        n_first_segment_tokens = np.sum(first_segment) - 2  # Remove [CLS], [SEP] token
        second_segment = np.logical_and(np.equal(segment_ids, 1), np.not_equal(input_ids, 0))
        n_second_segment_tokens = np.sum(second_segment) - 1  # Remove [SEP] token

        if self.target_segment == "first":
            n_q_seg_tokens = n_second_segment_tokens
            n_evi_seg_tokens = n_first_segment_tokens
            is_evidence_seg = first_segment
        elif self.target_segment == "second":
            n_q_seg_tokens = n_first_segment_tokens
            n_evi_seg_tokens = n_second_segment_tokens
            is_evidence_seg = second_segment
        else:
            raise ValueError()

        for i in range(len(input_ids)):
            is_evidence_seg[i] = is_evidence_seg[i] \
                                 and input_ids[i] != self.CLS_ID \
                                 and input_ids[i] != self.SEP_ID

        # 1 <= n_q_seg_tokens < ideal range < n_q_seg_tokens + 4 < n_evi_seg_tokens
        hard_max = n_evi_seg_tokens
        soft_max = n_q_seg_tokens + 4
        soft_min = n_q_seg_tokens
        hard_min = 1
        n_evidence_max = min(n_q_seg_tokens + 4, n_evi_seg_tokens)
        n_evidence_min = max(n_q_seg_tokens, 1)
        n_evidence_min = min(n_evidence_min, n_evidence_max)
        n_evidence_sel_size = random.randint(n_evidence_min, n_evidence_max)

        seq_len = len(segment_ids)
        if n_evidence_sel_size <= 0.5 * n_evi_seg_tokens:
            action_list: List[Action] = self._select_by_range(
                seq_len,
                is_evidence_seg,
                n_evidence_sel_size)
        else:
            action_list: List[Action] = self._select_by_range_delete(
                seq_len,
                is_evidence_seg,
                n_evidence_sel_size,
                n_evi_seg_tokens,
            )

        return action_list

    def _select_by_range(self, seq_len: int,
                         is_evidence_seg: List[int],
                         n_evidence_size: int) -> List[Action]:
        n_select_len = n_evidence_size
        action_list_as_range = self._enum_segment_range(is_evidence_seg,
                                                       seq_len, n_select_len)

        def set_value(st, ed) -> Action:
            return set_value_by_range(
                seq_len, st, ed,
                val_if_inside=1,
                val_if_outside=0,
            )

        action_list = [set_value(st, ed) for st, ed in action_list_as_range]
        return action_list

    def _select_by_range_delete(
            self, seq_len: int,
            is_evidence_seg: List[int],
            n_evidence_size,
            n_evi_seg_tokens) -> List[np.array]:
        n_delete = n_evi_seg_tokens - n_evidence_size
        if n_delete > 0:
            delete_range = self._enum_segment_range(is_evidence_seg,
                                                    seq_len, n_delete)
        elif n_delete == 0:
            delete_range = [(0, 0)]
        else:
            print('_select_by_range_delete')
            print(seq_len, is_evidence_seg, n_evidence_size, n_evi_seg_tokens)
            raise ValueError()

        def set_value(st, ed) -> Action:
            return set_value_by_range(
                seq_len, st, ed,
                val_if_inside=0,
                val_if_outside=1,
            )

        action_list = [set_value(st, ed) for st, ed in delete_range]
        return action_list

    def _enum_segment_range(self, is_evidence_seg: List[int], seq_len: int, select_len: int):
        action_list_as_range = []
        for i in range(seq_len):
            start_valid = is_evidence_seg[i]
            if not start_valid:
                continue

            start = i
            end = start + select_len
            while start < end:
                end_valid = end - 1 < seq_len
                end_valid = end_valid and is_evidence_seg[end - 1]
                if end_valid:
                    break
                else:
                    end = end - 1

            if start < end:
                action_list_as_range.append((start, end))
        return action_list_as_range

    def sample_actions(self, state_list: List[RLStateTensor]) -> List[List[Action]]:
        c_log.debug("SequenceLabelPolicyFunction sample_actions ENTRY")
        save_to_pickle(state_list, "state_list")
        state_tensor = batch_rl_state_list(state_list)
        c_log.debug("SequenceLabelPolicyFunction sample_actions Before model")
        evidence_score = self._get_token_scores(state_tensor)

        c_log.debug("SequenceLabelPolicyFunction sample_actions After model")  # 10 Sect
        n_sample_from_dist = int(self.epsilon * self.num_sample)
        n_sample_random = self.num_sample - n_sample_from_dist
        c_log.debug("SequenceLabelPolicyFunction sample_actions 2")
        per_state_action_samples: List[List[Action]] = []
        for i, state in enumerate(state_list):
            cand_action_list: List[Action] = self.enum_all_possible_action(state)
            dist_samples: List[Action] = self._get_dist_sample(
                cand_action_list, evidence_score[i:i + 1, :],
                n_sample_from_dist)  # [B, k, L]

            random_samples: List[Action] = self._get_random_sample(cand_action_list, n_sample_random)
            per_state: List[Action] = dist_samples + random_samples
            per_state_action_samples.append(per_state)

        c_log.debug("SequenceLabelPolicyFunction sample_actions EXIT")
        return per_state_action_samples

    def _get_token_scores(self, state_tensor):
        evidence_score = self.model(state_tensor)  # [B, L, 1]
        B, L, _ = get_shape_list2(evidence_score)
        evidence_score: np.array = tf.reshape(evidence_score, [B, L]).numpy()
        return evidence_score

    def _get_random_sample(self, action_candidates: np.array, n_random) -> List[List[int]]:
        action_indices = np.random.choice(len(action_candidates), [n_random])  # [k]
        sampled_actions: List[np.array] = [action_candidates[j] for j in action_indices]
        return sampled_actions

    def _get_dist_sample(self, action_candidates: np.array, evidence_score, k) -> List[Action]:
        # [m,] = [1, L] * [m, L]
        weight_for_action = np.sum(evidence_score * action_candidates, axis=1)
        probs_for_action = scipy.special.softmax(weight_for_action, axis=0)
        action_indices = np.random.choice(len(weight_for_action), [k], p=probs_for_action)  # [k]
        sampled_actions: List[np.array] = [action_candidates[j] for j in action_indices]
        return sampled_actions

    def get_mean_action(self, state_list: List[RLStateTensor]) -> List[Action]:
        c_log.debug("get_mean_action ENTRY")
        state_tensor = batch_rl_state_list(state_list)
        token_scores = self._get_token_scores(state_tensor)
        c_log.debug("get_mean_action 1")

        mean_action_list: List[Action] = []
        for i, state in enumerate(state_list):
            cand_action_list: List[Action] = self.enum_all_possible_action(state)
            evidence_score = token_scores[i:i + 1, :]
            weight_for_action = np.sum(evidence_score * cand_action_list, axis=1)
            idx = np.argmax(weight_for_action)
            mean_action_list.append(cand_action_list[idx])
        c_log.debug("get_mean_action EXIT")
        return mean_action_list

    # This would be called in gradient tape
    def get_log_action_prob(self,
                            state,  # e.g., input_ids, segment_ids
                            action_list: List[Action]
                            ) -> tf.Tensor:
        proba_dist = self.model(state)  # [B, L, 1]
        B, L, _ = get_shape_list2(proba_dist)
        proba_dist_ex = tf.reshape(proba_dist, [B, 1, L])  # [B, 1, L]
        action_list_f = tf.cast(action_list, tf.float32)  # [B, k, L]

        masked_prob = tf.multiply(proba_dist_ex, action_list_f)
        weights_for_each_action = tf.reduce_sum(masked_prob, axis=2)  # [B, K]
        probs_for_each_action = tf.nn.softmax(weights_for_each_action, axis=1)
        log_prob = tf.math.log(probs_for_each_action)
        return log_prob
