from abc import ABC, abstractmethod
from typing import Dict

import tensorflow as tf

import trainer_v2.per_project.transparency.mmp.probe.probe_common


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
    def get_eval_metrics(self) -> Dict[str, trainer_v2.per_project.transparency.mmp.probe.probe_common.Metric]:
        pass


