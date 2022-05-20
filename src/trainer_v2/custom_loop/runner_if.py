from abc import ABC, abstractmethod
from typing import Dict

import tensorflow as tf


class RunnerIF(ABC):
    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def get_keras_model(self) -> tf.keras.Model:
        pass

    @abstractmethod
    def do_init_checkpoint(self, init_checkpoint):
        pass

    @abstractmethod
    def train_step(self, item):
        pass

    @abstractmethod
    def loss_fn(self, labels, predictions):
        pass

    @abstractmethod
    def get_train_metrics(self) -> Dict[str, tf.keras.metrics.Metric]:
        pass

    @abstractmethod
    def get_eval_metrics(self) -> Dict[str, tf.keras.metrics.Metric]:
        pass

    # @abstractmethod
    # def build_dataset(self, input_path) -> tf.data.Dataset:
    #     pass

    @abstractmethod
    def set_keras_model(self, model):
        pass