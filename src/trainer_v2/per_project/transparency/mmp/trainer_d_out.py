from typing import Dict

import tensorflow as tf


from misc_lib import path_join
from trainer_v2.custom_loop.modeling_common.adam_decay import AdamWeightDecay
from trainer_v2.custom_loop.modeling_common.tf_helper import apply_gradient_warning_less
from trainer_v2.custom_loop.neural_network_def.inner_network import BertBasedModelIF
from trainer_v2.custom_loop.prediction_trainer import TrainerCommon, ModelV2IF
from trainer_v2.custom_loop.run_config2 import RunConfig2
from trainer_v2.custom_loop.train_loop_helper import fetch_metric_result
from trainer_v2.custom_loop.trainer_if import TrainerIFBase, EmptyEvalObject, EvalObjectIF


class EvalObject(EvalObjectIF):
    def __init__(self, model, eval_batches, strategy, log_metrics,
                 eval_steps=10):
        self.model = model
        self.eval_batches = eval_batches
        self.strategy = strategy
        self.eval_steps = eval_steps
        self.log_metrics = log_metrics
        self.metrics: Dict[str, tf.keras.metrics.Metric] = {}
        for key in log_metrics:
            self.metrics[key] = tf.keras.metrics.Mean(name=key)

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
            per_replica = self.strategy.run(self.eval_fn, args=args)

        step = self.model.optimizer.iterations
        for k in self.log_metrics:
            tf.summary.scalar("/eval/" + k, self.metrics[k].result(), step=step)
        eval_loss = self.metrics['loss'].result().numpy()
        return eval_loss, fetch_metric_result(self.metrics)

    @tf.function
    def eval_fn(self, item):
        model = self.model
        output_d = model(item, training=False)
        losses = output_d['loss']
        loss = tf.reduce_mean(losses)
        for k, v in output_d.items():
            if k in self.log_metrics:
                self.metrics[k].update_state(tf.reduce_mean(v))
        return loss


class TrainerDOut(TrainerCommon):
    def __init__(self, run_config: RunConfig2,
                 inner_model: ModelV2IF):
        super(TrainerDOut, self).__init__(run_config, inner_model)
        self.train_summary_writer = None
        self.train_metrics: Dict[str, tf.keras.metrics.Metric] = {}
        self.do_log = not run_config.device_config.use_tpu
        self.use_tpu = run_config.device_config.use_tpu

    def get_optimizer(self):
        return AdamWeightDecay(
            learning_rate=self.run_config.train_config.learning_rate,
            exclude_from_weight_decay=[]
        )

    def build_model(self):
        super(TrainerDOut, self).build_model()
        train_log_dir = path_join(self.run_config.train_config.model_save_path, "train_log")
        if self.do_log:
            if self.use_tpu:
                create_file_writer = tf.summary.experimental.create_file_writer
            else:
                create_file_writer = tf.summary.create_file_writer
            self.train_summary_writer = create_file_writer(train_log_dir, name="train")
            self.train_summary_writer.set_as_default()
        for key in self.inner_model.log_var:
            self.train_metrics[key] = tf.keras.metrics.Mean(name=key)

    def train_step(self, item):
        model = self.get_keras_model()
        with tf.GradientTape() as tape:
            output_d = model(item, training=True)
            losses = output_d['loss']
            loss = tf.reduce_mean(losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        step = self.optimizer.iterations
        for k, v in output_d.items():
            if k in self.inner_model.log_var:
                self.train_metrics[k].update_state(tf.reduce_mean(v))
                if self.do_log:
                    tf.summary.scalar(k, tf.reduce_mean(v), step=step)
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
        eval_object = EvalObject(
            self.model, eval_batches, strategy, self.inner_model.log_var,
                 100)
        return eval_object
