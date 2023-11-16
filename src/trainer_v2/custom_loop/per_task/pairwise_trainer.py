from abc import abstractmethod
from typing import Dict

import tensorflow as tf


from trainer_v2.custom_loop.evaler_if import EvalerIF
from trainer_v2.custom_loop.modeling_common.adam_decay import AdamWeightDecay
from trainer_v2.custom_loop.modeling_common.tf_helper import apply_gradient_warning_less
from trainer_v2.custom_loop.prediction_trainer import TrainerCommon, ModelV2IF
from trainer_v2.custom_loop.run_config2 import RunConfig2
from trainer_v2.custom_loop.trainer_if import EmptyEvalObject


class TrainerForLossReturningModel(TrainerCommon):
    def __init__(self, run_config: RunConfig2,
                 inner_model: ModelV2IF,
                 eval_object_factory=None
                 ):
        super(TrainerForLossReturningModel, self).__init__(run_config, inner_model)
        self.eval_object_factory = eval_object_factory

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
        if self.eval_object_factory is None:
            eval_object = EmptyEvalObject()
        else:
            eval_object = self.eval_object_factory(
                self.inner_model.get_keras_model(), eval_batches, strategy, 10
            )
        return eval_object



class PairwiseAcc(tf.keras.metrics.Metric):
    def __init__(self, **kwargs):
        super(PairwiseAcc, self).__init__(**kwargs)
        self.correct = self.add_weight(name='correct', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, predictions):
        pos_pred = predictions[:, 0, :]
        neg_pred = predictions[:, 1, :]
        is_correct = tf.less(neg_pred, pos_pred)
        is_correct_f = tf.cast(is_correct, tf.float32) # [B, 1]
        n_valid_correct = tf.reduce_sum(is_correct_f)
        self.correct.assign_add(n_valid_correct)
        self.count.assign_add(tf.reduce_sum(tf.ones_like(is_correct_f)))

    def result(self):
        return self.correct / self.count

    def reset_state(self):
        self.correct.assign(0.0)
        self.count.assign(0.0)


class PairwiseEvaler(EvalerIF):
    def __init__(self, run_config: RunConfig2):
        self.loss_mean = tf.keras.metrics.Mean(name="loss")
        self.pairwise_acc = PairwiseAcc()

    def eval_fn(self, item):
        model = self.get_keras_model()
        predictions, loss = model(item)
        self.loss_mean.update_state(loss)
        self.pairwise_acc.update_state(predictions)
        return loss

    def get_eval_metrics(self) -> Dict[str, tf.keras.metrics.Metric]:
        metrics = {}
        metrics["loss"] = self.loss_mean
        metrics["pairwise_acc"] = self.pairwise_acc
        return metrics

    