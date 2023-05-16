from typing import Dict

import tensorflow as tf

import trainer_v2.per_project.transparency.mmp.probe.probe_common
from trainer_v2.custom_loop.modeling_common.adam_decay import AdamWeightDecay
from trainer_v2.custom_loop.modeling_common.tf_helper import apply_gradient_warning_less
from trainer_v2.custom_loop.run_config2 import RunConfig2
from trainer_v2.custom_loop.trainer_if import TrainerIF


class RefModelTrainer(TrainerIF):
    def __init__(self, bert_params, model_config,
                 run_config: RunConfig2,
                 inner_model):
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
        self.training_loss = None
        self.train_metrics = None
        self.keras_model = None
        self.optimizer = None
        self.loss_fn_inner = None

    def parse_checkpoints(self):
        run_config = self.run_config
        bert_checkpoint, model_checkpoint = run_config.train_config.init_checkpoint.split(",")
        run_config.train_config.init_checkpoint = bert_checkpoint
        self.model_config.model_checkpoint = model_checkpoint

    def build_model(self):
        run_config = self.run_config

        if self.run_config.is_training():
            self.parse_checkpoints()
            model_checkpoint = self.model_config.model_checkpoint
            ref_model = tf.keras.models.load_model(model_checkpoint)
            for layer in ref_model.layers:
                if layer.name == "encoder1/bert":
                    ref_encoder1 = layer
                elif layer.name == "encoder2/bert":
                    ref_encoder2 = layer

            self.inner_model.build_model(self.bert_params, ref_encoder1, ref_encoder2, self.model_config)
            self.keras_model = self.inner_model.get_keras_model()
            self.keras_model.summary(140)
            optimizer = AdamWeightDecay(learning_rate=run_config.train_config.learning_rate)
            self.keras_model.optimizer = optimizer
            self.training_loss = tf.keras.metrics.Mean('training_loss', dtype=tf.float32)
            self.train_metrics = {'loss': self.training_loss}
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
        self.training_loss.update_state(loss)
        return loss

    def get_train_metrics(self) -> Dict[str, trainer_v2.per_project.transparency.mmp.probe.probe_common.Metric]:
        return self.train_metrics

    def get_eval_metrics(self) -> Dict[str, trainer_v2.per_project.transparency.mmp.probe.probe_common.Metric]:
        return self.eval_metrics

    def train_callback(self):
        try:
            self.inner_model.callback({'step': self.optimizer.iterations})
        except AttributeError:
            pass