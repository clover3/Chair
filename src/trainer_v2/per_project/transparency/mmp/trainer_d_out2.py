from typing import Dict

from cpath import output_path
from misc_lib import path_join

import tensorflow as tf
# from tensorflow.keras.metrics import Metric

from trainer_v2.custom_loop.modeling_common.adam_decay import AdamWeightDecay
from trainer_v2.custom_loop.modeling_common.tf_helper import apply_gradient_warning_less
from trainer_v2.custom_loop.prediction_trainer import TrainerCommon, ModelV2IF, ModelV3IF
from trainer_v2.custom_loop.run_config2 import RunConfig2
from trainer_v2.custom_loop.trainer_if import TrainerIFBase, EmptyEvalObject, EvalObjectIF
from typing import List, Iterable, Callable, Dict, Tuple, Set


def get_tf_log_dir(run_name):
    return path_join(output_path, "tf_log", run_name)


def get_tf_log_dir_val(run_name):
    return path_join(output_path, "tf_log", run_name + "_val")


Metric = tf.keras.metrics.Metric

class TrainerDOut2(TrainerIFBase):
    def __init__(self, run_config: RunConfig2,
                 inner_model: ModelV3IF):
        self.run_config = run_config
        self.eval_metrics = {}
        self.eval_metrics_factory = {}
        self.batch_size = run_config.common_run_config.batch_size
        self.inner_model: ModelV3IF = inner_model

        # These variables will be initialized by build_model()
        self.train_metrics = None
        self.model = None
        self.optimizer = None
        self.train_summary_writer = None
        self.train_metrics: Dict[str, Metric] = {}
        self.use_tpu = run_config.device_config.use_tpu
        self.do_log = not self.use_tpu

    def get_optimizer(self):
        return AdamWeightDecay(
            learning_rate=self.run_config.train_config.learning_rate,
            exclude_from_weight_decay=[]
        )

    def build_model(self):
        super(TrainerDOut2, self).build_model()
        run_name = self.run_config.common_run_config.run_name
        train_log_dir = get_tf_log_dir(run_name)
        eval_log_dir = get_tf_log_dir_val(run_name)
        self.inner_model.build_model(self.run_config)
        self.model = self.inner_model.get_keras_model()
        self.model.summary(140)
        self.train_metrics = self.inner_model.get_train_metrics()
        self.optimizer = self.get_optimizer()
        self.model.optimizer = self.optimizer

        self.train_metrics_summary = self.inner_model.get_train_metrics_for_summary()
        self.eval_metrics_summary = self.inner_model.get_eval_metrics_for_summary()
        if self.do_log:
            create_file_writer = tf.summary.create_file_writer
            self.train_summary_writer = create_file_writer(train_log_dir, name="train")
            self.eval_summary_writer = create_file_writer(eval_log_dir, name="eval")
        self.loss_fn = self.inner_model.get_loss_fn()

    def train_step(self, item):
        model = self.get_keras_model()
        with tf.GradientTape() as tape:
            output_d = model(item, training=True)
            loss = self.loss_fn(output_d)
        gradients = tape.gradient(loss, model.trainable_variables)
        apply_gradient_warning_less(self.optimizer, gradients, model.trainable_variables)

        step = self.optimizer.iterations
        if self.do_log:
            with self.train_summary_writer.as_default(step=step):
                for name, metric in self.train_metrics_summary.items():
                    metric.update_state(output_d)
                    self.make_summary_item(metric, name, step)
        return loss

    def make_summary_item(self, metric, name, step):
        tokens = name.split("/")
        if len(tokens) > 1:
            parent = tokens[0] + "/"
            tail = "/".join(tokens[1:])
            with tf.name_scope(parent):
                tf.summary.scalar(tail, metric.result(), step=step)
        else:
            tf.summary.scalar(name, metric.result(), step=step)

    def do_init_checkpoint(self, init_checkpoint):
        self.inner_model.init_checkpoint(init_checkpoint)

    def set_keras_model(self, model):
        self.model = model

    def get_keras_model(self):
        return self.model

    def get_train_metrics(self) -> Dict[str, tf.keras.metrics.Metric]:
        return {}

    def get_eval_metrics(self) -> Dict[str, tf.keras.metrics.Metric]:
        return self.eval_metrics

    def train_callback(self):
        try:
            self.inner_model.callback({'step': self.optimizer.iterations})
        except AttributeError:
            pass

    def get_eval_object(self, eval_batches, strategy):
        self.eval_batches = eval_batches
        self.strategy = strategy
        do_eval_fn = self.do_eval
        class EvalObject(EvalObjectIF):
            def do_eval(self):
                return do_eval_fn()
        eval_object = EvalObject()
        return eval_object

    def do_eval(self):
        model = self.get_keras_model()

        for m in self.eval_metrics.values():
            m.reset_state()

        @tf.function
        def eval_fn(item):
            output_d = model(item, training=False)
            loss = self.loss_fn(output_d)
            for name, metric in self.eval_metrics_summary.items():
                metric.update_state(output_d)
            return loss

        max_step = sum(1 for _ in self.eval_batches)
        slice_step = max_step

        iterator = iter(self.eval_batches)
        total_loss = 0.0
        n_step = 0.

        for idx in range(slice_step):
            args = next(iterator),
            per_replica_losses = self.strategy.run(eval_fn, args=args)
            loss = self.strategy.reduce(
                tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
            total_loss += loss
            n_step += 1.

        avg_loss = total_loss / n_step

        step = self.optimizer.iterations
        with self.eval_summary_writer.as_default(step=step):
            for name, metric in self.eval_metrics_summary.items():
                self.make_summary_item(metric, name, step)

        return avg_loss, {}
