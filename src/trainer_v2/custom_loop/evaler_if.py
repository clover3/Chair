from abc import ABC, abstractmethod
from typing import Dict

import tensorflow as tf




class EvalerIF(ABC):
    def set_model(self, model):
        self.model = model

    def get_keras_model(self) -> tf.keras.Model:
        return self.model

    @abstractmethod
    def eval_fn(self, item):
        pass

    @abstractmethod
    def get_eval_metrics(self) -> Dict[str, tf.keras.metrics.Metric]:
        pass


