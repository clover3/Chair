###
# Routines for Training / Loading model / Inference
#
#
#
from typing import Dict

import numpy as np
import tensorflow as tf

from data_generator.bert_input_splitter import get_sep_loc
from data_generator.tokenizer_wo_tf import JoinEncoder
from tlm.qtype.partial_relevance.attention_based.hard_concrete_optimize.attention_opt import \
    TransformerAttentionInfModel
from tlm.qtype.partial_relevance.attention_based.hard_concrete_optimize.attnetion_opt_utils import Tensor
from tlm.qtype.partial_relevance.attention_based.hard_concrete_optimize.segment_helper import get_always_active_mask, \
    get_always_active_mask_qt
from tlm.qtype.partial_relevance.attention_based.hard_concrete_optimize.train_utils import session_run_print, \
    load_model_by_dir, print_dict_float_values
from tlm.qtype.partial_relevance.eval_data_structure import SegmentedInstance
from trainer.tf_train_module_v2 import init_session


def get_train_op_from_grads_and_tvars(grads, tvars, lr):
    print("lr", lr)
    optimizer = tf.keras.optimizers.Adam(lr)
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
    train_op = optimizer.apply_gradients(zip(grads, tvars))
    return train_op


def merge_tensor_dict(tensor_dict: Dict[str, Tensor], item: Dict[str, np.array]) -> Dict[Tensor, np.array]:
    out_d = {}
    for k, v in item.items():
        tensor = tensor_dict[k]
        out_d[tensor] = np.expand_dims(v, 0)
    return out_d


class AttentionMaskOptimizer:
    def __init__(self, model, hp):
        self.task = model
        gradients = tf.gradients(self.task.loss, self.task.log_alpha)
        self.train_op = get_train_op_from_grads_and_tvars(gradients, [self.task.log_alpha], hp.lr)
        self.sess = init_session()
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def load_model(self, save_dir):
        variables = tf.compat.v1.trainable_variables()
        variables_to_restore = [v for v in variables if v.name != "log_alpha:0"]
        assert len(variables) == len(variables_to_restore) + 1
        load_model_by_dir(self.sess, save_dir, variables_to_restore)

    def train(self, item: Dict, num_steps):
        fetch_tensors_d = {
            'loss': self.task.loss,
            'logits': self.task.logits,
            'loss_1': self.task.loss_1,
            'loss_2': self.task.loss_2,
            'train_op': self.train_op,
            'sample_accuracy': self.task.sample_accuracy,
        }
        fetch_tensors_d.update(self.task.get_debug_vars())
        fetch_tensors_keys = list(fetch_tensors_d.keys())

        for step in range(num_steps):
            fetch_tensors_list = [fetch_tensors_d[k] for k in fetch_tensors_keys]
            feed_dict = merge_tensor_dict(self.task.x_dict, item)
            fetched_values = self.sess.run(fetch_tensors_list,
                                           feed_dict=feed_dict)
            fetched_values_d = dict(zip(fetch_tensors_keys, fetched_values))
            fetched_values_d['step'] = step
            print_dict_float_values(fetched_values_d)
            self.task.debug_print(fetched_values_d)

    def fetch_inf_mask(self, item: Dict) -> np.array:
        feed_dict = merge_tensor_dict(self.task.x_dict, item)
        inf_pred_mask, = self.sess.run([self.task.inf_predicted_mask], feed_dict=feed_dict)
        return inf_pred_mask[0]


def init_model_for_inference(hp, save_dir):
    tf.compat.v1.reset_default_graph()
    task = TransformerAttentionInfModel(hp)
    sess = init_session()
    sess.run(tf.compat.v1.global_variables_initializer())
    load_model_by_dir(sess, save_dir)
    return task, sess


def inference(sess, task, item):
    fetch_tensors_d = {
        'loss': task.loss,
        'loss_1': task.loss_1,
        'loss_2': task.loss_2,
        'logits': task.logits,
        'base_logits': task.base_logits,
        'masked_logits': task.masked_logits,
        'sample_accuracy': task.sample_accuracy,
    }
    feed_dict = merge_tensor_dict(task.x_dict, item)
    return session_run_print(sess, feed_dict, fetch_tensors_d)


class AttnOptEncoderWrap:
    def __init__(self, max_seq_length):
        self.join_encoder = JoinEncoder(max_seq_length)
        self.max_seq_length = max_seq_length

    def encode(self, inst: SegmentedInstance) -> Dict:
        x0, x1, x2 = self.join_encoder.join(inst.text1.tokens_ids, inst.text2.tokens_ids)
        x3 = get_always_active_mask(inst, self.max_seq_length)
        item = {
            "input_ids": x0,
            "input_mask": x1,
            "segment_ids": x2,
            "always_active_mask": x3
        }
        return item


def get_linear_valid_target_mask(input_ids):
    idx_sep1, idx_sep2 = get_sep_loc(input_ids)
    linear_valid_target_mask = np.zeros_like(input_ids, dtype=np.float)
    for i in range(idx_sep1 + 1, idx_sep2):
        linear_valid_target_mask[i] = 1
    return linear_valid_target_mask


class AttnOptQTEncoder:
    def __init__(self, max_seq_length):
        self.max_seq_length = max_seq_length
        self.join_encoder = JoinEncoder(max_seq_length)

    def encode(self, inst: SegmentedInstance, target_idx) -> Dict:
        x0, x1, x2 = self.join_encoder.join(inst.text1.tokens_ids, inst.text2.tokens_ids)
        x3 = get_always_active_mask_qt(inst, self.max_seq_length, target_idx)
        item = {
            "input_ids": x0,
            "input_mask": x1,
            "segment_ids": x2,
            "always_active_mask": x3
        }
        input_ids = item['input_ids']
        linear_valid_target_mask = get_linear_valid_target_mask(input_ids)
        item['linear_valid_target_mask'] = linear_valid_target_mask
        return item
