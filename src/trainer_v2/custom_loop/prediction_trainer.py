from abc import ABC, abstractmethod
from typing import Dict

import tensorflow as tf

from trainer_v2.custom_loop.modeling_common.tf_helper import apply_gradient_warning_less
from trainer_v2.custom_loop.run_config2 import RunConfig2
from trainer_v2.custom_loop.trainer_if import TrainerIF



# Purpose of ModelV3IF: to define custom init checkpoint functions
#   build_model: defines keras model
class ModelV3IF(ABC):
    @abstractmethod
    def build_model(self, run_config):
        pass

    @abstractmethod
    def get_keras_model(self) -> tf.keras.models.Model:
        pass

    @abstractmethod
    def init_checkpoint(self, model_path):
        pass


class PredictionTrainerCommon(TrainerIF):
    def __init__(self, run_config: RunConfig2,
                 inner_model: ModelV3IF):
        self.run_config = run_config
        self.eval_metrics = {}
        self.eval_metrics_factory = {}
        self.batch_size = run_config.common_run_config.batch_size
        self.inner_model = inner_model

        # These variables will be initialized by build_model()
        self.train_metrics = None
        self.model = None
        self.optimizer = None

    @abstractmethod
    def get_optimizer(self):
        pass

    @abstractmethod
    def loss_fn(self, labels, predictions):
        pass

    def build_model(self):
        if self.run_config.is_training():
            self.inner_model.build_model(self.run_config)
            self.model = self.inner_model.get_keras_model()

            self.model.summary(140)
            self.train_metrics = {}
            self.optimizer = self.get_optimizer()
            self.model.optimizer = self.optimizer
        else:
            pass
        for k, v in self.eval_metrics_factory.items():
            self.eval_metrics[k] = v()

    def do_init_checkpoint(self, init_checkpoint):
        self.inner_model.init_checkpoint(init_checkpoint)

    def set_keras_model(self, model):
        self.model = model

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