from typing import Dict

import tensorflow as tf

from trainer_v2.partial_processing2.RunConfig2 import RunConfig2
from trainer_v2.partial_processing2.modeling_common.bert_common import BERT_CLS, ModelConfig
from trainer_v2.partial_processing2.modeling_common.tf_helper import apply_gradient_warning_less
from trainer_v2.partial_processing2.runner_if import RunnerIF
from trainer_v2.train_util.input_fn_common import create_dataset_common


class NLIRunner(RunnerIF):
    def __init__(self, bert_params, model_config, run_config: RunConfig2):
        self.bert_params = bert_params
        self.model_config = model_config
        self.run_config = run_config
        self.eval_metrics = {}

        self.eval_metrics_factory = {
            'acc': lambda :tf.keras.metrics.Accuracy(name='accuracy', dtype=None)
        }
        self.batch_size = run_config.common_run_config.batch_size

    def build_model(self):
        run_config = self.run_config
        if self.run_config.is_training():
            bert_cls = BERT_CLS(self.bert_params, self.model_config)
            self.bert_cls = bert_cls
            self.keras_model = bert_cls.model
            self.init_loss()
            optimizer = tf.optimizers.Adam(learning_rate=run_config.train_config.learning_rate)
            bert_cls.model.optimizer = optimizer
            self.optimizer = optimizer
        else:
            pass
        for k, v in self.eval_metrics_factory.items():
            self.eval_metrics[k] = v()
        self.loss_fn_inner = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    def set_keras_model(self, model):
        self.keras_model = model

    def init_loss(self):
        self.loss_fn_inner = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    def loss_fn(self, labels, predictions):
        per_example_loss = self.loss_fn_inner(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.batch_size)

    def get_keras_model(self):
        return self.keras_model

    def get_model_ref_for_ckpt(self):
        return self.bert_cls

    def train_step(self, item):
        model = self.get_keras_model()
        x1, x2, y = item
        with tf.GradientTape() as tape:
            prediction = model([x1, x2], training=True)
            loss = self.loss_fn(y, prediction)

        # c_log.debug("train_cls called")
        gradients = tape.gradient(loss, model.trainable_variables)
        apply_gradient_warning_less(self.optimizer, gradients, model.trainable_variables)
        return loss

    def get_metrics(self) -> Dict[str, tf.keras.metrics.Metric]:
        return self.eval_metrics

    def get_dataset(self, input_files):
        return get_nli_data(input_files, self.run_config, self.model_config)


def get_nli_data(file_path, run_config: RunConfig2, model_config: ModelConfig) -> tf.data.Dataset:
    seq_length = model_config.max_seq_length

    def decode_record(record):
        name_to_features = {
            'input_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
            'input_mask': tf.io.FixedLenFeature([seq_length], tf.int64),
            'segment_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
            'label_ids': tf.io.FixedLenFeature([], tf.int64),
        }
        return tf.io.parse_single_example(record, name_to_features)

    def _select_data_from_record(record):
        for k, v in record.items():
            record[k] = tf.cast(v, tf.int32)
        entry = record['input_ids'], record['segment_ids'], record['label_ids']
        return entry
    return create_dataset_common(_select_data_from_record, run_config.common_run_config.batch_size,
                                 decode_record, file_path, run_config.is_training())