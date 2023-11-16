import tensorflow as tf
from tensorflow.python.distribute.distribute_lib import Strategy

from trainer_v2.custom_loop.dataset_factories import get_qd_multi_seg_dataset
from trainer_v2.custom_loop.modeling_common.network_utils import split_stack_input
from trainer_v2.custom_loop.train_loop_helper import fetch_metric_result
from trainer_v2.custom_loop.trainer_if import EvalObjectIF


class EEEvalObject(EvalObjectIF):
    def __init__(self, model_config, model, eval_batches, dist_strategy: Strategy,
                 eval_steps=10):
        self.loss = tf.keras.metrics.Mean(name='dev_loss')
        self.eval_batches = eval_batches
        self.model = model
        self.dist_strategy: Strategy = dist_strategy
        self.eval_steps = eval_steps
        self.model_config = model_config
        self.accuracy = tf.keras.metrics.Accuracy(name='accuracy', dtype=None)
        self.metrics = {"accuracy": self.accuracy}

    @tf.function
    def eval_fn(self, item):
        q_seq_len = self.model_config.max_num_qt * self.model_config.segment_len
        d_seq_len = self.model_config.max_num_dt * self.model_config.segment_len

        prediction, loss = self.model(item, training=False)
        self.loss.update_state(loss)
        label_scores = item['scores']
        label_scores_stack = tf.reshape(label_scores, [-1, self.model_config.max_num_qt, self.model_config.max_num_dt])
        q_input_ids = item["q_input_ids"]
        d_input_ids = item["d_input_ids"]
        valid_f = get_is_valid_mask(q_input_ids, d_input_ids, q_seq_len, d_seq_len, self.model_config.segment_len)
        valid_i = tf.cast(valid_f, tf.int32)

        thres = 1.1
        binary_pred = tf.cast(tf.less(thres, prediction), tf.int32)
        binary_label = tf.cast(tf.less(thres, label_scores_stack), tf.int32)
        self.accuracy.update_state(binary_label, binary_pred, valid_i)

    def do_eval(self):
        for m in self.metrics.values():
            m.reset_state()

        max_step = sum(1 for _ in self.eval_batches)

        if self.eval_steps >= 0:
            slice_step = self.eval_steps
        else:
            slice_step = max_step

        iterator = iter(self.eval_batches)
        for idx in range(slice_step):
            args = next(iterator),
            per_replica = self.dist_strategy.run(self.eval_fn, args=args)

        eval_loss = self.loss.result().numpy()
        metrics = self.metrics
        metric_res = fetch_metric_result(metrics)
        return eval_loss, metric_res


def get_qd_multi_seg_dataset_from_model_config(
        input_files, is_for_training, model_config, run_config):
    q_seq_len = model_config.max_num_qt * model_config.segment_len
    d_seq_len = model_config.max_num_dt * model_config.segment_len
    n_scores = model_config.max_num_qt * model_config.max_num_dt
    dataset = get_qd_multi_seg_dataset(
        input_files, run_config,
        q_seq_len, d_seq_len, n_scores,
        is_for_training)
    return dataset


def get_is_valid_mask(q_input_ids, d_input_ids, q_seq_len, d_seq_len, segment_len):
    q_input_ids_stacked = split_stack_input([q_input_ids], q_seq_len, segment_len)[0]
    d_input_ids_stacked = split_stack_input([d_input_ids], d_seq_len, segment_len)[0]

    def is_valid_entry(input_ids_stack):
        return tf.logical_not(tf.reduce_all(tf.equal(input_ids_stack, 0), axis=2))

    q_is_valid = tf.cast(is_valid_entry(q_input_ids_stacked), tf.float32)
    d_is_valid = tf.cast(is_valid_entry(d_input_ids_stacked), tf.float32)
    is_valid_arr = tf.expand_dims(q_is_valid, axis=2) * tf.expand_dims(d_is_valid, axis=1)
    return tf.cast(is_valid_arr, tf.float32)