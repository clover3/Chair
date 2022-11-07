import random
from typing import List
import numpy as np
import tensorflow as tf

from data_generator2.segmented_enc.es.evidence_candidate_gen import pool_delete_indices
from misc_lib import ceil_divide
from models.transformer.bert_common_v2 import get_shape_list2
from trainer_v2.evidence_selector.evidence_candidates import get_st_ed
from trainer_v2.reinforce.monte_carlo_policy_function import PolicyFunction, Action
from trainer_v2.evidence_selector.seq_pred_policy_gradient import RLStateTensor, batch_rl_state_list


def delete_seg1(input_ids, segment_ids) -> List[int]:
    seg2_st, _ = get_st_ed(segment_ids)
    seg1_tokens = input_ids[1:seg2_st-1]
    n_tokens = len(seg1_tokens)
    g = 0.5
    g_inv = int(1 / g)
    max_del = ceil_divide(n_tokens, g_inv)
    num_del = random.randint(1, max_del)
    indices: List[int] = pool_delete_indices(num_del, n_tokens, g)
    items = [1 for _ in input_ids]
    for i in indices:
        items[i+1] = 0

    return items


class SequenceLabelPolicyFunction(PolicyFunction):
    # State : input_ids, segment_ids
    def __init__(self, model, k=10):
        self.model = model
        self.k = k
        self.epsilon = 0.5

    def dummy_model(self, item):
        input_ids, segment_ids = item
        B = len(input_ids.numpy())
        L = len(input_ids[0])
        t1 = np.random.random([B, L])
        ret = np.stack([t1, 1-t1], axis=2)
        return tf.constant(ret)

    def sample_actions(self, state_list: List[RLStateTensor]) -> List[List[Action]]:
        y_s_list = []
        for _ in range(self.k):
            if random.random() < self.epsilon:
                y_s = self._get_random_sample(state_list)
            else:
                y_s = self._get_dist_sample(state_list)

            y_s_list.append(y_s)

        t = tf.stack(y_s_list, axis=1)

        per_state_action_samples: List[List[Action]] = []
        for i in range(len(state_list)):
            per_state = []
            for j in range(self.k):
                action: tf.Tensor = t[i, j]
                per_state.append(action)
            per_state_action_samples.append(per_state)
        return per_state_action_samples

    def _get_random_sample(self, state_list: List[RLStateTensor]) -> List[List[int]]:
        def state_to_sample(state: RLStateTensor):
            return delete_seg1(state.input_ids, state.segment_ids)

        # Return: [B, K, L]
        return list(map(state_to_sample, state_list))

    def _get_dist_sample(self, state_list: List[RLStateTensor]) -> List[List[int]]:
        state_tensor = batch_rl_state_list(state_list)
        proba_dist = self.dummy_model(state_tensor)  # [B, L, 2]
        B, L, _ = get_shape_list2(proba_dist)
        samples = tf.random.categorical(tf.reshape(proba_dist, [-1, 2]), 1)
        samples = tf.reshape(samples, [B, L])  # [Binary]
        return samples

    def get_mean_action(self, state_list: List[RLStateTensor]) -> Action:
        state_tensor = batch_rl_state_list(state_list)
        proba_dist = self.dummy_model(state_tensor)  # [B, L, 2]
        return tf.argmax(proba_dist, axis=2)

    # This would be called in gradient tape
    def get_log_action_prob(self,
                            state,  # e.g., input_ids, segment_ids
                            action_list  # e.g., [B, K, L]
                            ) -> tf.Tensor:
        # [Batch, Number of Samples]
        proba_dist = self.model(state)  # [B, L]
        proba_dist_ex = tf.expand_dims(proba_dist, 1)
        action_list = tf.one_hot(action_list, depth=2)
        action_list_f = tf.cast(action_list, tf.float32)
        masked_prob = tf.multiply(proba_dist_ex, action_list_f)
        prob_for_each_dim = tf.reduce_sum(masked_prob, axis=3)
        return tf.reduce_sum(tf.math.log(prob_for_each_dim))
