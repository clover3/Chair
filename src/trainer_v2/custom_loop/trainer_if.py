from abc import ABC, abstractmethod
from typing import Dict

import tensorflow as tf
Metric = tf.keras.metrics.Metric


class TrainerIFBase(ABC):
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
    def get_train_metrics(self) -> Dict[str, Metric]:
        pass

    @abstractmethod
    def get_eval_metrics(self) -> Dict[str, Metric]:
        pass

    @abstractmethod
    def set_keras_model(self, model):
        pass

    @abstractmethod
    def train_callback(self):
        pass

    @abstractmethod
    def get_eval_object(self, batches, strategy):
        pass


class TrainerIF(TrainerIFBase):
    @abstractmethod
    def loss_fn(self, labels, predictions):
        pass


class EvalObjectIF:
    @abstractmethod
    def do_eval(self):
        pass


class EmptyEvalObject(EvalObjectIF):
    def do_eval(self):
        return 0, {}
