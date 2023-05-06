import logging
import os

from tensorflow.python.ops.summary_ops_v2 import create_file_writer
from cpath import output_path
from misc_lib import path_join
from trainer_v2.custom_loop.modeling_common.adam_decay import AdamWeightDecay

from trainer_v2.custom_loop.modeling_common.tf_helper import distribute_dataset
from trainer_v2.train_util.get_tpu_strategy import get_strategy2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
from trainer_v2.chair_logging import c_log, IgnoreFilter, IgnoreFilterRE
import tensorflow as tf

from cpath import get_bert_config_path
from taskman_client.wrapper3 import report_run3
from trainer_v2.custom_loop.prediction_trainer import ModelV2IF, ModelV3IF
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2
from trainer_v2.per_project.transparency.mmp.tt_model.model_conf_defs import InputShapeConfigTT, InputShapeConfigTT100_4
from trainer_v2.train_util.arg_flags import flags_parser

class LinearV3:
    def __init__(self):
        x = tf.keras.layers.Input(shape=(2,), dtype=tf.int32, name="x")
        x_f = tf.cast(x, tf.float32)
        y = tf.keras.layers.Dense(1)(x_f)
        inputs = [x,]
        output = {'pred': y}
        self.model = tf.keras.models.Model(inputs=inputs, outputs=output)

    def get_metrics(self) :
        output_d = {}
        metric = ProbeMAE("mae")
        output_d["mae2"] = metric
        return output_d



Metric = tf.keras.metrics.Metric
class ProbeMAE(Metric):
    def __init__(self, name, **kwargs):
        super(ProbeMAE, self).__init__(name=name, **kwargs)
        self.mae = self.add_weight(name='mae', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        # self.metric_inner = tf.keras.metrics.MeanAbsoluteError()

    def update_state(self, output_d, _sample_weight=None):
        v = tf.reduce_sum(output_d['pred'])
        self.mae.assign_add(v)
        self.count.assign_add(1.0)

    def result(self):
        return self.mae / self.count

    def reset_state(self):
        self.mae.assign(0.0)
        self.count.assign(0.0)


class LinearModel(ModelV3IF):
    def __init__(self, input_shape: InputShapeConfigTT):
        self.inner_model = None
        self.model: tf.keras.models.Model = None
        self.loss = None
        self.input_shape: InputShapeConfigTT = input_shape
        self.log_var = ["loss"]

    def build_model(self):
        self.inner_model = LinearV3()

    def get_keras_model(self) -> tf.keras.models.Model:
        return self.inner_model.model

    def init_checkpoint(self, init_checkpoint):
        pass

    def get_train_metrics(self):
        return {}

    def get_train_metrics_for_summary(self):
        return self.inner_model.get_metrics()

    def get_loss_fn(self):
        def get_loss(d):
            return tf.reduce_sum(d['pred'])
        return get_loss


@report_run3
def main(args):
    c_log.info(__file__)
    run_config: RunConfig2 = get_run_config2(args)
    run_config.print_info()
    input_shape = InputShapeConfigTT100_4()
    model_v2 = LinearModel(input_shape)
    optimizer = AdamWeightDecay(
        learning_rate=1e-3,
        exclude_from_weight_decay=[]
    )


    def build_dataset(input_files, is_for_training):
        def generator():
            for _ in range(100):
                yield [0., 0.]
        train_dataset = tf.data.Dataset.from_generator(
            generator,
            output_types=(tf.float32),
            output_shapes=(tf.TensorShape([2])))


        return train_dataset.batch(2)


    strategy = get_strategy2(False, "")

    train_dataset = build_dataset(run_config.dataset_config.train_files_path, True)
    eval_dataset = build_dataset(run_config.dataset_config.eval_files_path, False)
    dist_train_dataset = distribute_dataset(strategy, train_dataset)
    eval_batches = distribute_dataset(strategy, eval_dataset)
    train_log_dir = path_join(output_path, "train_log")
    step_idx = 0
    with strategy.scope():
        model_v2.build_model()
        train_summary_writer = create_file_writer(train_log_dir, name="train")
        train_summary_writer.set_as_default()
        train_metrics = model_v2.get_train_metrics_for_summary()

        def train_step(item):
            model = model_v2.get_keras_model()
            with tf.GradientTape() as tape:
                output_d = model(item, training=True)

            step = optimizer.iterations
            for name, metric in train_metrics.items():
                metric.update_state(output_d)
                sc = tf.summary.scalar(name, metric.result(), step=step)
                print(sc)
            return tf.constant(0.0)

        @tf.function
        def distributed_train_step(train_itr, steps_per_execution):
            # try:
            total_loss = 0.0
            n_step = 0.
            for _ in tf.range(steps_per_execution):
                batch_item = next(train_itr)
                per_replica_losses = strategy.run(train_step, args=(batch_item, ))
                loss = strategy.reduce(
                    tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
                total_loss += loss
                n_step += 1.

            train_loss = total_loss / n_step
            return train_loss
        train_itr = iter(dist_train_dataset)
        for m in train_metrics.values():
            m.reset_state()

        train_loss = distributed_train_step(train_itr, 1)
        step_idx += 1



if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)


