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
    def get_model_ref_for_ckpt(self):
        pass

    @abstractmethod
    def train_step(self, item):
        pass

    @abstractmethod
    def loss_fn(self, labels, predictions):
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, tf.keras.metrics.Metric]:
        pass

    @abstractmethod
    def get_dataset(self, input_path) -> tf.data.Dataset:
        pass

    @abstractmethod
    def init_loss(self):
        pass

    @abstractmethod
    def set_keras_model(self, model):
        pass