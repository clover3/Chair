from typing import Dict

import tensorflow as tf

from trainer_v2.custom_loop.modeling_common.adam_decay import AdamWeightDecay
from trainer_v2.custom_loop.modeling_common.tf_helper import apply_gradient_warning_less
from trainer_v2.custom_loop.run_config2 import RunConfig2
from trainer_v2.custom_loop.trainer_if import TrainerIF


class TrainerHuggingfaceInit(TrainerIF):
    def __init__(self, model_config,
                 run_config: RunConfig2,
                 model_factory):
        self.model_config = model_config
        self.run_config = run_config
        self.eval_metrics = {}
        self.eval_metrics_factory = {
            'acc': lambda: tf.keras.metrics.Accuracy(name='accuracy', dtype=None)
        }
        self.batch_size = run_config.common_run_config.batch_size
        self.model_factory = model_factory

        # These variables will be initialized by build_model()
        self.train_metrics = None
        self.optimizer = None
        self.loss_fn_inner = None

    def build_model(self):
        if self.run_config.is_training():
            self.model = self.model_factory()
            self.model.summary(140)
            self.train_metrics = {}
            self.optimizer = self.model.optimizer
        else:
            pass
        for k, v in self.eval_metrics_factory.items():
            self.eval_metrics[k] = v()

    def set_keras_model(self, model):
        self.model = model

    def loss_fn(self, labels, predictions):
        per_example_loss = self.loss_fn_inner(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.batch_size)

    def get_keras_model(self):
        return self.model

    def train_step(self, item):
        model = self.get_keras_model()
        x, y = item
        with tf.GradientTape() as tape:
            prediction = model(x, training=True)
            loss = self.loss_fn(y, prediction)

        gradients = tape.gradient(loss, model.trainable_variables)
        apply_gradient_warning_less(self.optimizer, gradients, model.trainable_variables)
        return loss

    def get_train_metrics(self) -> Dict[str, tf.keras.metrics.Metric]:
        return self.train_metrics

    def get_eval_metrics(self) -> Dict[str, tf.keras.metrics.Metric]:
        return self.eval_metrics

    def train_callback(self):
        try:
            self.model.callback({'step': self.optimizer.iterations})
        except AttributeError:
            pass

    def get_eval_object(self, eval_batches, strategy):
        return None
