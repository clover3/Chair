from trainer_v2.chair_logging import c_log
import os
from typing import List, Iterable, Callable, Dict, Tuple, Set, Any
from transformers import TFBertMainLayer

from tf_util.lib.tf_funcs import find_layer
from trainer_v2.custom_loop.train_loop_helper import get_strategy_from_config, eval_tensor
from trainer_v2.per_project.transparency.mmp.alignment.debugs.model_load_dev import get_dev_batch
from trainer_v2.per_project.transparency.mmp.probe.probe_common import build_paired_inputs_concat
from trainer_v2.per_project.transparency.mmp.trainer_d_out2 import TrainerDOut2
from trainer_v2.per_project.transparency.mmp.trnsfmr_util import get_dummy_input_for_bert_layer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from trainer_v2.per_project.transparency.mmp.probe.probe_network import ProbeOnBERT, ProbeLossFromDict

import sys
import tensorflow as tf

from trainer_v2.custom_loop.prediction_trainer import ModelV3IF
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2
from trainer_v2.custom_loop.trainer_if import TrainerIFBase
from trainer_v2.per_project.transparency.mmp.tt_model.model_conf_defs import InputShapeConfigTT, InputShapeConfigTT100_4
from trainer_v2.train_util.arg_flags import flags_parser


class ProbeModel(ModelV3IF):
    def __init__(self, input_shape: InputShapeConfigTT):
        self.network = None
        self.model: tf.keras.models.Model = None
        self.loss = None
        self.input_shape: InputShapeConfigTT = input_shape

    def build_model(self, run_config):
        init_checkpoint = run_config.train_config.init_checkpoint
        c_log.info("Loading model from {}".format(init_checkpoint))
        ranking_model = tf.keras.models.load_model(init_checkpoint, compile=False)
        self.network = ProbeOnBERT(ranking_model)
        self.loss = ProbeLossFromDict()

        print("layers")
        for l in self.network.model.layers:
            print(l.name)

    def get_keras_model(self) -> tf.keras.models.Model:
        return self.network.model

    def init_checkpoint(self, init_checkpoint):
        pass

    def get_train_metrics(self):
        return {}

    def get_train_metrics_for_summary(self):
        return self.network.get_probe_metrics()

    def get_loss_fn(self):
        return self.loss


def get_probe_like(bert_model):
    bert_main_layer_ckpt = find_layer(bert_model, "bert")
    classifier_ckpt = find_layer(bert_model, "classifier")
    bert_main_layer_ckpt_param = bert_main_layer_ckpt.get_weights()
    classifier_ckpt_param = classifier_ckpt.get_weights()

    bert_config = bert_main_layer_ckpt._config

    # bert_main_layer = build_bert_layer(bert_main_layer_ckpt)
    bert_main_layer = TFBertMainLayer(bert_config, name="bert")
    _ = bert_main_layer(get_dummy_input_for_bert_layer())
    bert_main_layer.set_weights(bert_main_layer_ckpt_param)

    classifier = tf.keras.layers.Dense(1)
    hidden_size = bert_config.hidden_size
    _ = classifier(tf.zeros([1, hidden_size]))
    classifier.set_weights(classifier_ckpt_param)

    c_log.info("identify_layers")
    keys = ["input_ids", "token_type_ids"]
    input_concat_d, inputs = build_paired_inputs_concat(keys)

    input_ids = input_concat_d["input_ids"]
    segment_ids = input_concat_d["token_type_ids"]

    c_log.info("bert_main_layer")
    # bert_input = [input_ids, segment_ids]
    bert_input = {
        "input_ids": input_ids,
        "token_type_ids": segment_ids
    }
    bert_outputs = bert_main_layer(
        bert_input,
        output_attentions=True,
        output_hidden_states=True)
    logits = classifier(bert_outputs.pooler_output)
    logits = tf.stop_gradient(logits, "logits_stop_gradient")
    probe_model_output: Dict[str, Any] = {
        "logits": logits,
    }
    model = tf.keras.models.Model(inputs=inputs, outputs=probe_model_output)
    return model


# @report_run3
def main(args):
    c_log.info(__file__)
    run_config: RunConfig2 = get_run_config2(args)
    run_config.print_info()
    input_shape = InputShapeConfigTT100_4()
    model_v3 = ProbeModel(input_shape)
    trainer: TrainerIFBase = TrainerDOut2(run_config, model_v3)

    strategy = get_strategy_from_config(run_config)
    with strategy.scope():
        init_checkpoint = run_config.train_config.init_checkpoint
        c_log.info("Loading model from {}".format(init_checkpoint))
        paired_model = tf.keras.models.load_model(init_checkpoint, compile=False)
        c_log.info("Loading model Done")
        old_bert_layer = find_layer(paired_model, "bert")

        dense_layer = find_layer(paired_model, "classifier")
        new_bert_layer = TFBertMainLayer(old_bert_layer._config, name="bert")
        param_values = tf.keras.backend.batch_get_value(old_bert_layer.weights)
        _ = new_bert_layer(get_dummy_input_for_bert_layer())
        tf.keras.backend.batch_set_value(zip(new_bert_layer.weights, param_values))
        input_ids1 = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_ids1")
        segment_ids1 = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="token_type_ids1")
        input_ids2 = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_ids2")
        segment_ids2 = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="token_type_ids2")
        inputs = [input_ids1, segment_ids1, input_ids2, segment_ids2]
        input_1 = {
            'input_ids': input_ids1,
            'token_type_ids': segment_ids1,
        }

        bert_output = new_bert_layer(input_1)
        logits = dense_layer(bert_output['pooler_output'])[:, 0]
        single_ranking_model = tf.keras.models.Model(inputs=inputs, outputs=logits)

        batch = get_dev_batch()
        ranking_model_output = single_ranking_model(batch)
        print("ranking_model_output", ranking_model_output)

        # network = ProbeOnBERT(paired_model)
        # print("layers")
        # for l in network.model.layers:
        #     print(l.name)
        # trainer.build_model()
        # model = trainer.get_keras_model()
        model = get_probe_like(paired_model)

        output = model(batch)
        print("logits", output['logits'])

        # saver = ModelSaver(
        #     model, run_config.train_config.model_save_path, get_current_step
        # )
        # saver.save()
        # model.save_weights("/tmp/tmp_weights")


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)


