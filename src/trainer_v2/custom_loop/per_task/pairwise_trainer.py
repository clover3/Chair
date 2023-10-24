from abc import abstractmethod
from typing import Dict

import tensorflow as tf


from trainer_v2.custom_loop.evaler_if import EvalerIF
from trainer_v2.custom_loop.modeling_common.adam_decay import AdamWeightDecay
from trainer_v2.custom_loop.modeling_common.tf_helper import apply_gradient_warning_less
from trainer_v2.custom_loop.prediction_trainer import TrainerCommon, ModelV2IF
from trainer_v2.custom_loop.run_config2 import RunConfig2
from trainer_v2.custom_loop.trainer_if import EmptyEvalObject


class PairwiseTrainer(TrainerCommon):
    def __init__(self, run_config: RunConfig2,
                 inner_model: ModelV2IF):
        super(PairwiseTrainer, self).__init__(run_config, inner_model)

    def get_optimizer(self):
        return AdamWeightDecay(
            learning_rate=self.get_learning_rate(),
            exclude_from_weight_decay=[]
        )

    def train_step(self, item):
        model = self.get_keras_model()
        with tf.GradientTape() as tape:
            predictions, loss = model(item, training=True)

        gradients = tape.gradient(loss, model.trainable_variables)
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


class EvaluatorIF:
    @abstractmethod
    def load_model(self, model_path):
        pass

    @abstractmethod
    def get_eval_metrics(self):
        pass

    @abstractmethod
    def eval_fn(self, args):
        pass


class PairwiseEvaler(EvalerIF):
    def __init__(self, run_config: RunConfig2):
        self.eval_metrics = NotImplemented

    def eval_fn(self, item):
        model = self.get_keras_model()
        predictions, loss = model(item)
        for m in self.eval_metrics.values():
            m.update_state(predictions, loss)

        return loss

    def get_eval_metrics(self) -> Dict[str, tf.keras.metrics.Metric]:
        return self.eval_metrics

    