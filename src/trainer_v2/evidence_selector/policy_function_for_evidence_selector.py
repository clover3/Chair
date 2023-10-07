import random
from typing import List
import numpy as np
import tensorflow as tf

from cache import save_to_pickle
from data_generator2.segmented_enc.es_nli.evidence_candidate_gen import pool_delete_indices
from misc_lib import ceil_divide, tensor_to_list
from models.transformer.bert_common_v2 import get_shape_list2
from trainer_v2.chair_logging import c_log
from trainer_v2.evidence_selector.evidence_candidates import get_st_ed
from trainer_v2.reinforce.monte_carlo_policy_function import PolicyFunction, Action
from trainer_v2.evidence_selector.seq_pred_policy_gradient import batch_rl_state_list
from trainer_v2.evidence_selector.defs import RLStateTensor


def delete_seg1(total_len, seg2_st):
    n_tokens = seg2_st - 2
    g = 0.5
    g_inv = int(1 / g)
    max_del = ceil_divide(n_tokens, g_inv)
    num_del = random.randint(1, max_del)
    indices: List[int] = pool_delete_indices(num_del, n_tokens, g)
    items = [1] * total_len
    for i in indices:
        items[i + 1] = 0
    return items


class SequenceLabelPolicyFunction(PolicyFunction):
    # State : input_ids, segment_ids
    def __init__(self, model, num_sample=10, valid_seg="first"):
        self.model = model
        self.num_sample = num_sample
        self.epsilon = 0.5
        self.valid_seg = valid_seg

    def __dummy_model(self, item):
        input_ids, segment_ids = item
        B = len(input_ids.numpy())
        L = len(input_ids[0])
        t1 = np.random.random([B, L])
        ret = np.stack([t1, 1-t1], axis=2)
        return tf.constant(ret)

    def sample_actions(self, state_list: List[RLStateTensor]) -> List[List[Action]]:
        c_log.debug("SequenceLabelPolicyFunction sample_actions ENTRY")
        save_to_pickle(state_list, "state_list")
        state_tensor = batch_rl_state_list(state_list)
        c_log.debug("SequenceLabelPolicyFunction sample_actions Before model")
        proba_dist = self.model(state_tensor)  # [B, L, 2]
        c_log.debug("SequenceLabelPolicyFunction sample_actions After model")  # 10 Sect
        n_dist = int(self.epsilon * self.num_sample)
        n_random = self.num_sample - n_dist

        c_log.debug("SequenceLabelPolicyFunction sample_actions Samples {} dist".format(n_dist))
        dist_sample = self._get_dist_sample(proba_dist, n_dist)
        c_log.debug("SequenceLabelPolicyFunction sample_actions 2")

        per_state_action_samples: List[List[Action]] = []
        for i, state in enumerate(state_list):
            samples: List[Action] = self._get_random_sample(state, n_random)
            per_state: List[Action] = samples
            per_state.extend(dist_sample[i])
            per_state_action_samples.append(per_state)

        c_log.debug("SequenceLabelPolicyFunction sample_actions EXIT")
        return per_state_action_samples

    def _get_random_sample(self, state: RLStateTensor, n_random) -> List[List[int]]:
        seg2_st, _ = get_st_ed(state.segment_ids_np)
        action_list = []
        for _ in range(n_random):
            action = delete_seg1(len(state.input_ids), seg2_st)
            action_list.append(action)
        return action_list

    def _get_dist_sample(self, proba_dist, k) -> List[List[int]]:
        B, L, _ = get_shape_list2(proba_dist)
        samples = tf.random.categorical(tf.reshape(proba_dist, [-1, 2]), k)
        samples = tf.reshape(samples, [B, L, k])  # [Binary]
        return tf.transpose(samples, [0, 2, 1])

    def get_mean_action(self, state_list: List[RLStateTensor]) -> List[List[int]]:
        state_tensor = batch_rl_state_list(state_list)
        proba_dist = self.model(state_tensor)  # [B, L, 2]
        return tf.argmax(proba_dist, axis=2)

    def get_top_k_action(self, state_list: List[RLStateTensor]) -> Action:
        state_tensor = batch_rl_state_list(state_list)
        proba_dist = self.model(state_tensor)  # [B, L, 2]
        action_list = []
        for idx, s in enumerate(state_list):
            n_seg2 = tf.reduce_sum(s.segment_ids)
            scores = proba_dist[idx, :, 1]
            is_first_seg = tf.logical_and(tf.equal(s.segment_ids, 0), tf.not_equal(s.input_ids, 0))
            # n_sel = tf.cast(tf.reduce_sum(tf.cast(is_first_seg, tf.float32)) * 0.4, tf.int32)
            n_sel = n_seg2
            scores = scores * tf.cast(is_first_seg, tf.float32)
            sel_indices = tf.argsort(scores)[::-1][:n_sel]
            action = [0 for _ in range(len(s.input_ids))]
            for k in sel_indices:
                action[k] = 1
            action_list.append(action)
        return action_list

    # This would be called in gradient tape
    def get_log_action_prob(self,
                            state,  # e.g., input_ids, segment_ids
                            action_list  # e.g., [B, K, L]
                            ) -> tf.Tensor:
        # [Batch, Number of Samples]
        input_ids, segment_ids = state
        if self.valid_seg == "first":
            is_valid_action = tf.logical_and(tf.equal(segment_ids, 0), tf.not_equal(input_ids, 0))
        elif self.valid_seg == "second":
            is_valid_action = tf.logical_and(tf.equal(segment_ids, 1), tf.not_equal(input_ids, 0))
        else:
            raise ValueError()
        eps = 1e-9

        proba_dist = self.model(state)  # [B, L]
        proba_dist_ex = tf.expand_dims(proba_dist, 1)  # [B, 1, L, 2]
        action_list = tf.one_hot(action_list, depth=2)  #  [B, K, L, 2]
        action_list_f = tf.cast(action_list, tf.float32)
        masked_prob = tf.multiply(proba_dist_ex, action_list_f)
        prob_for_each_dim = tf.reduce_sum(masked_prob, axis=3)  # [B, K, L]
        is_valid_action_mask = tf.cast(tf.expand_dims(is_valid_action, axis=1), tf.float32)  # [B, 1, L]
        log_prob = tf.math.log(prob_for_each_dim)
        log_prob = tf.multiply(log_prob, is_valid_action_mask)
        n_valid = tf.reduce_sum(is_valid_action_mask, axis=2) + eps #[B, 1]
        log_prob_sum = tf.reduce_sum(log_prob, axis=2)  # [ B, K]
        ret = tf.divide(log_prob_sum, n_valid) # [ B, K]
        return ret
