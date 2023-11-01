import sys
from typing import List, Iterable, Callable, Dict, Tuple, Set

from cpath import get_bert_config_path
from taskman_client.wrapper3 import report_run3
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.dataset_factories import get_classification_dataset_hf_to_bert_f2, create_dataset_common
from trainer_v2.custom_loop.definitions import ModelConfig512_2, ModelConfigType
from trainer_v2.custom_loop.modeling_common.adam_decay import AdamWeightDecay
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config
from trainer_v2.custom_loop.modeling_common.tf_helper import distribute_dataset
from trainer_v2.custom_loop.neural_network_def.inner_network import BertBasedModelIF
from trainer_v2.custom_loop.neural_network_def.two_seg_alt import TwoSegConcatLogitCombine, CombineByLogitAdd
from trainer_v2.custom_loop.per_task.trainer import Trainer
from trainer_v2.custom_loop.run_config2 import get_run_config2_train, RunConfig2
from trainer_v2.custom_loop.train_loop import tf_run, tf_run2
from trainer_v2.custom_loop.train_loop_helper import get_strategy_from_config, eval_tensor
from trainer_v2.custom_loop.trainer_if import TrainerIF, TrainerIFBase
from trainer_v2.train_util.arg_flags import flags_parser
import tensorflow as tf
import numpy as np
from tensorflow import keras


feature_size = 1


def get_constant_dataset(
        file_path,
        run_config: RunConfig2,
        is_for_training,
    ) -> tf.data.Dataset:
    def decode_record(record):
        value = np.random.random([feature_size]).astype(np.float32)
        return tf.constant([1.0]), tf.constant(0, dtype=tf.int32)

    return create_dataset_common(decode_record, run_config,
                                 file_path, is_for_training)


class GetLinearRegression(BertBasedModelIF):
    def __init__(self):
        super(GetLinearRegression, self).__init__()

    def build_model(self, _bert_params, _config: ModelConfigType):
        input_feature = tf.keras.layers.Input(shape=(1,), dtype='float32',
                                              name="feature")

        dense = tf.keras.layers.Dense(1, use_bias=False)
        pred = dense(input_feature)
        output = tf.expand_dims(pred, axis=-1)
        print(output)
        inputs = (input_feature, )
        model = keras.Model(inputs=inputs, outputs=output, name="bert_model")
        self.model: keras.Model = model

    def get_keras_model(self):
        return self.model

    def init_checkpoint(self, init_checkpoint):
        pass


class Trainer2(Trainer):
    def build_model(self):
        run_config = self.run_config
        if self.run_config.is_training():
            self.inner_model.build_model(self.bert_params, self.model_config)
            self.keras_model = self.inner_model.get_keras_model()
            self.keras_model.summary(140)

            decay_steps = run_config.train_config.train_step
            lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
                run_config.train_config.learning_rate,
                decay_steps,
                end_learning_rate=0,
                power=1.0,
                cycle=False,
                name=None
            )
            optimizer = AdamWeightDecay(learning_rate=lr_schedule,
                                        exclude_from_weight_decay=[],
                                        )
            self.keras_model.optimizer = optimizer
            self.train_metrics = {}
            self.optimizer = optimizer
        else:
            pass
        for k, v in self.eval_metrics_factory.items():
            self.eval_metrics[k] = v()
        self.loss_fn_inner = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)


def tf_run_train_debug(run_config: RunConfig2,
                 trainer: TrainerIFBase,
                 dataset_factory: Callable[[str, bool], tf.data.Dataset]
                 ):
    c_log.debug("tf_run_train ENTRY")
    strategy = get_strategy_from_config(run_config)
    c_log.debug("tf_run_inner initializing dataset")
    train_dataset = dataset_factory(run_config.dataset_config.train_files_path, True)
    eval_dataset = dataset_factory(run_config.dataset_config.eval_files_path, False)
    dist_train_dataset = distribute_dataset(strategy, train_dataset)
    eval_batches = distribute_dataset(strategy, eval_dataset)
    c_log.debug("Building models")
    with strategy.scope():
        trainer.build_model()
        c_log.info("Loading checkpoints: {}".format(run_config.train_config.init_checkpoint))
        trainer.do_init_checkpoint(run_config.train_config.init_checkpoint)

        model = trainer.get_keras_model()
        dense_layer = model.layers[1]
        print(dense_layer)
        weights = dense_layer.weights[0]
        print(weights)

        eval_object = trainer.get_eval_object(eval_batches, strategy)

        train_itr = iter(dist_train_dataset)

        current_step = eval_tensor(model.optimizer.iterations)
        c_log.info("Current step = {}".format(current_step))
        conf_steps_per_execution = run_config.common_run_config.steps_per_execution

        @tf.function
        def distributed_train_step(train_itr, steps_per_execution):
            # try:
            total_loss = 0.0
            n_step = 0.
            for _ in tf.range(steps_per_execution):
                batch_item = next(train_itr)
                per_replica_losses = strategy.run(trainer.train_step, args=(batch_item, ))
                loss = strategy.reduce(
                    tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
                total_loss += loss
                n_step += 1.

            train_loss = total_loss / n_step
            return train_loss
        step_idx = current_step
        last_weight = 0
        c_log.info("START Training")
        while step_idx < run_config.train_config.train_step:
            current_step = eval_tensor(model.optimizer.iterations)
            c_log.debug("Current step = {}".format(current_step))
            cur_weight = eval_tensor(weights)
            diff = last_weight - cur_weight
            print(step_idx, cur_weight, diff)
            last_weight = cur_weight

            if step_idx == 0:
                steps_to_execute = 1
            elif step_idx % conf_steps_per_execution > 0:
                steps_to_execute = conf_steps_per_execution - step_idx % conf_steps_per_execution
            else:
                steps_to_execute = conf_steps_per_execution
            c_log.debug("Execute {} steps".format(steps_to_execute))
            train_loss = distributed_train_step(train_itr, steps_to_execute)

            trainer.get_keras_model()
            step_idx += steps_to_execute
            c_log.debug("step_idx={}".format(step_idx))
            trainer.train_callback()
        c_log.info("Training completed")


@report_run3
def main(args):
    c_log.info(__file__)
    run_config: RunConfig2 = get_run_config2_train(args)
    run_config.train_config.train_step = 1000
    run_config.print_info()

    bert_params = load_bert_config(get_bert_config_path())
    model_config = ModelConfig512_2()
    task_model = GetLinearRegression()
    trainer: TrainerIF = Trainer2(bert_params, model_config, run_config, task_model)

    def build_dataset(input_files, is_for_training):
        return get_constant_dataset(input_files, run_config, is_for_training)

    tf_run_train_debug(run_config, trainer, build_dataset)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)


