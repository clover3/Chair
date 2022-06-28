import code
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from cpath import get_bert_config_path
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config
from trainer_v2.custom_loop.neural_network_def.ctx2 import CtxChunkInteraction
from trainer_v2.custom_loop.neural_network_def.segmented_enc import FuzzyLogicLayerM


import sys
import tensorflow as tf

from cache import save_to_pickle
from taskman_client.wrapper3 import report_run3
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.dataset_factories import get_two_seg_data
from trainer_v2.custom_loop.neural_network_def.siamese import ModelConfig2SegProject
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2_nli
from trainer_v2.train_util.arg_flags import flags_parser


def save_layer_cache(layer_no, layer_val):
    save_to_pickle(layer_val, "layer_{}_debug_var".format(layer_no))


def get_model_init(model_config):
    inner = CtxChunkInteraction(FuzzyLogicLayerM)
    c_log.info("Using CtxChunkInteraction")
    bert_params = load_bert_config(get_bert_config_path())
    inner.build_model(bert_params, model_config)
    return inner


def plot_is_valid_decision(args):
    c_log.info("Start {}".format(__file__))
    c_log.setLevel(logging.DEBUG)
    run_config: RunConfig2 = get_run_config2_nli(args)
    model_config = ModelConfig2SegProject()
    def dataset_factory(input_files, is_for_training):
        return get_two_seg_data(input_files, run_config, model_config, is_for_training)

    model_path = run_config.eval_config.model_save_path
    c_log.debug("Building model")
    inner = get_model_init(model_config)
    model = inner.model

    c_log.debug("tf_run_inner initializing dataset")
    eval_dataset = dataset_factory(run_config.dataset_config.eval_files_path, False)
    per_layer_outputs = []
    for layer in model.layers:
        per_layer_outputs.append(layer.output)

    c_log.debug("Building new model...")
    new_model = tf.keras.Model(inputs=model.inputs, outputs=inner.is_valid_decision)
    ce = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    output = new_model.predict(eval_dataset, steps=1)
    print(output)


@report_run3
def main(args):
    c_log.info("Start {}".format(__file__))
    c_log.setLevel(logging.DEBUG)
    run_config: RunConfig2 = get_run_config2_nli(args)
    model_config = ModelConfig2SegProject()
    def dataset_factory(input_files, is_for_training):
        return get_two_seg_data(input_files, run_config, model_config, is_for_training)

    model_path = run_config.eval_config.model_save_path
    c_log.debug("Building model")
    inner = get_model_init(model_config)
    model = inner.model

    c_log.debug("tf_run_inner initializing dataset")
    eval_dataset = dataset_factory(run_config.dataset_config.eval_files_path, False)
    per_layer_outputs = []
    for layer in model.layers:
        per_layer_outputs.append(layer.output)

    c_log.debug("Building new model...")
    new_model = tf.keras.Model(inputs=model.inputs, outputs=per_layer_outputs)
    ce = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    itr = iter(eval_dataset)
    item = next(itr)
    x, y = item
    c_log.debug("Running forward run")
    with tf.GradientTape() as tape:
        prediction = model(x, training=True)
        loss = ce(y, prediction)

    tvars = model.trainable_variables
    grads = tape.gradient(loss, tvars)
    for g, v in zip(grads, tvars):
        any_nan = tf.reduce_any(tf.math.is_nan(v))
        print(g, "There is nan :", any_nan.numpy())
    code.interact(local=locals())


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)


