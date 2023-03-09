from typing import Dict

import tensorflow as tf
from tensorflow.python.distribute.distribute_lib import Strategy

from trainer_v2.custom_loop.modeling_common.adam_decay import AdamWeightDecay
from trainer_v2.custom_loop.modeling_common.tf_helper import apply_gradient_warning_less
from trainer_v2.custom_loop.neural_network_def.inner_network import ClassificationModelIF
from trainer_v2.custom_loop.run_config2 import RunConfig2
from trainer_v2.custom_loop.train_loop_helper import fetch_metric_result
from trainer_v2.custom_loop.trainer_if import TrainerIF


class ClassificationEvalObject:
    def __init__(self, model, eval_batches, dist_strategy: Strategy,
                 loss_fn,
                 eval_metrics,
                 eval_steps=10):
        self.loss = tf.keras.metrics.Mean(name='dev_loss')
        self.metrics: Dict[str, tf.keras.metrics.Metric] = eval_metrics
        self.eval_batches = eval_batches
        self.model = model
        self.dist_strategy: Strategy = dist_strategy
        self.loss_fn = loss_fn
        self.eval_steps = eval_steps

    @tf.function
    def eval_fn(self, item):
        x, y = item
        prediction = self.model(x, training=False)
        loss = self.loss_fn(y, prediction)
        self.loss.update_state(loss)
        pred = tf.argmax(prediction, axis=1)
        for m in self.metrics.values():
            m.update_state(y, pred)

    def do_eval(self):
        for m in self.metrics.values():
            m.reset_state()

        max_step = sum(1 for _ in self.eval_batches)

        if self.eval_steps >= 0:
            slice_step = self.eval_steps
        else:
            slice_step = max_step

        iterator = iter(self.eval_batches)
        for idx in range(slice_step):
            args = next(iterator),
            per_replica = self.dist_strategy.run(self.eval_fn, args=args)

        eval_loss = self.loss.result().numpy()
        metrics = self.metrics
        metric_res = fetch_metric_result(metrics)
        return eval_loss, metric_res


def get_precision_recall_factories():
    return {
        'precision': lambda: tf.keras.metrics.Precision(name='precision', dtype=None),
        'recall': lambda: tf.keras.metrics.Recall(name='recall', dtype=None),
    }


def get_tp_fp_tn_fn_factories():
    return {
        'TrueNegatives': lambda: tf.keras.metrics.TrueNegatives(name='TrueNegatives', dtype=None),
        'FalseNegatives': lambda: tf.keras.metrics.FalseNegatives(name='FalseNegatives', dtype=None),
        'TruePositives': lambda: tf.keras.metrics.TruePositives(name='TruePositives', dtype=None),
        'FalsePositives': lambda: tf.keras.metrics.FalsePositives(name='FalsePositives', dtype=None),
    }


class Trainer(TrainerIF):
    def __init__(self, bert_params, model_config,
                 run_config: RunConfig2,
                 inner_model: ClassificationModelIF):
        self.bert_params = bert_params
        self.model_config = model_config
        self.run_config = run_config
        self.eval_metrics = {}
        self.eval_metrics_factory = {
            'acc': lambda: tf.keras.metrics.Accuracy(name='accuracy', dtype=None)
        }
        self.batch_size = run_config.common_run_config.batch_size
        self.inner_model = inner_model

        # These variables will be initialized by build_model()
        self.train_metrics = None
        self.keras_model = None
        self.optimizer = None
        self.loss_fn_inner = None

    def build_model(self):
        run_config = self.run_config
        if self.run_config.is_training():
            self.inner_model.build_model(self.bert_params, self.model_config)
            self.keras_model = self.inner_model.get_keras_model()
            self.keras_model.summary(140)
            optimizer = AdamWeightDecay(learning_rate=run_config.train_config.learning_rate,
                                        exclude_from_weight_decay=[],
                                        )
            self.keras_model.optimizer = optimizer
            self.train_metrics = {}
            self.optimizer = optimizer
        else:
            pass
        for k, v in self.eval_metrics_factory.items():
            self.eval_metrics[k] = v()
        self.loss_fn_inner = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    def set_keras_model(self, model):
        self.keras_model = model

    def loss_fn(self, labels, predictions):
        per_example_loss = self.loss_fn_inner(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.batch_size)

    def get_keras_model(self):
        return self.keras_model

    def do_init_checkpoint(self, init_checkpoint):
        return self.inner_model.init_checkpoint(init_checkpoint)

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
            self.inner_model.callback({'step': self.optimizer.iterations})
        except AttributeError:
            pass

    def get_eval_object(self, eval_batches, strategy):
        eval_object = ClassificationEvalObject(self.get_keras_model(),
                                               eval_batches,
                                               strategy,
                                               self.loss_fn,
                                               self.get_eval_metrics())
        return eval_object
