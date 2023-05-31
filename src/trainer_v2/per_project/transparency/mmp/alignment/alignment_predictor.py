from typing import Iterable, Dict, List, Tuple

import numpy as np
import tensorflow as tf

from misc_lib import TimeEstimator
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.train_loop_helper import get_strategy_from_config
from trainer_v2.per_project.transparency.mmp.alignment.grad_extractor import GradExtractor, ModelEncoded


class AlignmentPredictor:
    def __init__(self, run_config, compute_alignment_fn):
        strategy = get_strategy_from_config(run_config)
        c_log.info("{}".format(strategy))
        self.strategy = strategy
        self.compute_alignment_fn = compute_alignment_fn
        with strategy.scope():
            self.extractor = GradExtractor(
                run_config.predict_config.model_save_path,
                run_config.common_run_config.batch_size,
                strategy
            )

    def predict_for_qd_iter(self, qd_itr, num_record) -> Iterable[Dict]:
        ticker = TimeEstimator(num_record)
        with self.strategy.scope():
            me_itr: Iterable[ModelEncoded] = self.extractor.encode(qd_itr)
            for me in me_itr:
                aligns = self.compute_alignment_fn(me)
                logits = me.logits.tolist()
                out_info = {'logits': logits, 'aligns': aligns}
                ticker.tick()
                yield out_info


def get_seg2_start(token_type_ids):
    for i in range(len(token_type_ids)):
        if token_type_ids[i] == 1:
            return i
    return len(token_type_ids)


def get_idx_of(input_ids, token_id):
    for idx, t in enumerate(input_ids):
        if t == token_id:
            return idx
    return -1


def compute_alignment_for_taget_q_word_id(
        item: ModelEncoded,
        target_q_word_id: int,
        top_k=10,
) -> List[Tuple[int, int, float]]:
    d_start = get_seg2_start(item.token_type_ids)
    idx = get_idx_of(item.input_ids, target_q_word_id)
    # print("Target is {} at {} th".format(input_tokens[idx], idx))
    # print("input_tokens", input_tokens)
    assert item.token_type_ids[idx] == 0

    attn_mul_grad = item.attentions * item.attention_grads

    t = tf.reduce_sum(attn_mul_grad, axis=0)  # average over layers
    t = tf.reduce_sum(t, axis=0)  # average over heads
    t = t[:, idx] + t[idx, :]  #
    # t = t[idx, :] #
    rank = np.argsort(t)[::-1]
    rank = [j for j in rank if item.input_ids[j] != 0 and j >= d_start]
    output: List[Tuple[int, int, float]] = []
    for j in rank[:top_k]:
        score = t[j].numpy().tolist()
        output.append((int(j), int(item.input_ids[j]), float(score)))
    return output


def compute_alignment_first_layer(
        item: ModelEncoded,
        target_q_word_id: int,
        top_k=10,
) -> List[Tuple[int, int, float]]:
    d_start = get_seg2_start(item.token_type_ids)
    idx = get_idx_of(item.input_ids, target_q_word_id)
    # print("Target is {} at {} th".format(input_tokens[idx], idx))
    # print("input_tokens", input_tokens)
    assert item.token_type_ids[idx] == 0

    attn_mul_grad = item.attentions * item.attention_grads

    t = attn_mul_grad[0]  # Select first layer
    t = tf.reduce_sum(t, axis=0)  # average over heads
    t = t[:, idx] + t[idx, :]  #
    # t = t[idx, :] #
    rank = np.argsort(t)[::-1]
    rank = [j for j in rank if item.input_ids[j] != 0 and j >= d_start]
    output: List[Tuple[int, int, float]] = []
    for j in rank[:top_k]:
        score = t[j].numpy().tolist()
        output.append((int(j), int(item.input_ids[j]), float(score)))
    return output


def compute_alignment_any_pair(
        item: ModelEncoded,
        top_k=10,
) -> List[Tuple[int, int, float]]:
    d_start = get_seg2_start(item.token_type_ids)

    SEP_ID = 102
    def iter_query_indices():
        # iterate through idx which is query segment, select save all pairs
        for i in range(1, len(item.input_ids)):
            if item.token_type_ids[i] == 0 and item.input_ids[i] != SEP_ID:
                yield i
            else:
                break

    query_indices: Iterable[int] = iter_query_indices()
    attn_mul_grad = item.attentions * item.attention_grads
    t = tf.reduce_sum(attn_mul_grad, axis=0)  # average over layers
    t = tf.reduce_sum(t, axis=0)  # average over heads
    output: List[Tuple[int, int, float]] = []

    for idx in query_indices:
        q_term = int(item.input_ids[idx])
        t_reduced = t[:, idx] + t[idx, :]  #
        # t = t[idx, :] #
        rank = np.argsort(t_reduced)[::-1]
        rank = [j for j in rank if item.input_ids[j] != 0 and j >= d_start]
        for j in rank[:top_k]:
            score = t_reduced[j].numpy().tolist()
            d_term = int(item.input_ids[j])
            output.append((q_term, d_term, float(score)))
    return output

