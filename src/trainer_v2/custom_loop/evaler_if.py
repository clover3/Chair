from abc import ABC, abstractmethod
from typing import Dict

import tensorflow as tf


class EvalerIF(ABC):
    @abstractmethod
    def build(self, model):
        pass

    @abstractmethod
    def get_keras_model(self) -> tf.keras.Model:
        pass

    @abstractmethod
    def eval_fn(self, item):
        pass

    @abstractmethod
    def get_eval_metrics(self) -> Dict[str, tf.keras.metrics.Metric]:
        pass


