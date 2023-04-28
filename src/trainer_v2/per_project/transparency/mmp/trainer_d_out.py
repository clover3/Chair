from typing import Dict

import tensorflow as tf

from misc_lib import path_join
from trainer_v2.custom_loop.modeling_common.adam_decay import AdamWeightDecay
from trainer_v2.custom_loop.modeling_common.tf_helper import apply_gradient_warning_less
from trainer_v2.custom_loop.neural_network_def.inner_network import BertBasedModelIF
from trainer_v2.custom_loop.prediction_trainer import TrainerCommon, ModelV2IF
from trainer_v2.custom_loop.run_config2 import RunConfig2
from trainer_v2.custom_loop.trainer_if import TrainerIFBase, EmptyEvalObject


class TrainerDOut(TrainerCommon):
    def __init__(self, run_config: RunConfig2,
                 inner_model: ModelV2IF):
        super(TrainerDOut, self).__init__(run_config, inner_model)
        self.train_summary_writer = None

    def get_optimizer(self):
        return AdamWeightDecay(
            learning_rate=self.run_config.train_config.learning_rate,
            exclude_from_weight_decay=[]
        )

    def build_model(self):
        super(TrainerDOut, self).build_model()
        train_log_dir = path_join(self.run_config.train_config.model_save_path, "train_log")
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir, name="train")
        self.train_summary_writer.set_as_default()

    def train_step(self, item):
        model = self.get_keras_model()
        with tf.GradientTape() as tape:
            output_d = model(item, training=True)
            losses = output_d['loss']
            loss = tf.reduce_mean(losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        step = self.optimizer.iterations
        for k, v in output_d.items():
            if k in self.inner_model.log_var:
                tf.summary.scalar(k, tf.reduce_mean(v), step=step)
        tf.summary.scalar('constant', 1.0, step=step)
        apply_gradient_warning_less(self.optimizer, gradients, model.trainable_variables)
        return loss

    def get_train_metrics(self) -> Dict[str, tf.keras.metrics.Metric]:
        return self.train_metrics

    def get_eval_metrics(self) -> Dict[str, tf.keras.metrics.Metric]:
        return self.eval_metrics

    def train_callback(self):
        try:
            self.inner_model.callback({'step': self.optimizer.iterations})
        except AttributeError:
            pass

    def get_eval_object(self, eval_batches, strategy):
        eval_object = EmptyEvalObject()
        return eval_object
