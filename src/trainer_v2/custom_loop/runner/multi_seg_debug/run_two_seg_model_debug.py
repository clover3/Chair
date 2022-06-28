import code
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from cpath import get_bert_config_path
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config
from trainer_v2.custom_loop.neural_network_def.ctx2 import CtxSingle
from trainer_v2.custom_loop.neural_network_def.segmented_enc import FuzzyLogicLayerM


import sys
import tensorflow as tf

from cache import save_to_pickle
from taskman_client.wrapper3 import report_run3
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.dataset_factories import get_two_seg_data
from trainer_v2.custom_loop.modeling_common.tf_helper import distribute_dataset
from trainer_v2.custom_loop.neural_network_def.siamese import ModelConfig2SegProject
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2_nli
from trainer_v2.custom_loop.train_loop_helper import get_strategy_from_config
from trainer_v2.train_util.arg_flags import flags_parser


def save_layer_cache(layer_no, layer_val):
    save_to_pickle(layer_val, "layer_{}_debug_var".format(layer_no))


def get_model_init(model_config):
    inner = CtxSingle(FuzzyLogicLayerM)
    bert_params = load_bert_config(get_bert_config_path())
    inner.build_model(bert_params, model_config)
    return inner.model

# Check variables

@report_run3
def main(args):
    c_log.info("Start {}".format(__file__))
    c_log.setLevel(logging.DEBUG)
    run_config: RunConfig2 = get_run_config2_nli(args)
    model_config = ModelConfig2SegProject()

    def dataset_factory(input_files, is_for_training):
        return get_two_seg_data(input_files, run_config, model_config, is_for_training)

    strategy = get_strategy_from_config(run_config)
    c_log.debug("strategy %s" % strategy)
    with strategy.scope():
        c_log.debug("Loading model")
        model_path = run_config.eval_config.model_save_path
        model = get_model_init(model_config)
        # model = tf.keras.models.load_model(model_path)

    c_log.debug("tf_run_inner initializing dataset")
    eval_dataset = dataset_factory(run_config.dataset_config.eval_files_path, False)
    eval_dataset = distribute_dataset(strategy, eval_dataset)
    per_layer_outputs = []
    for layer in model.layers:
        # if not isinstance(layer, tf.keras.layers.InputLayer):
        per_layer_outputs.append(layer.output)

    c_log.debug("Model predicts...")
    with strategy.scope():
        new_model = tf.keras.Model(inputs=model.inputs, outputs=per_layer_outputs)
        outputs = new_model.predict(eval_dataset, steps=1)
    c_log.debug("...Done")

    for idx, (layer_output, layer) in enumerate(zip(outputs, per_layer_outputs)):
        print(idx, layer.name)

    for idx, (layer_output, layer) in enumerate(zip(outputs, per_layer_outputs)):
        print(idx, layer.name, layer_output)

    code.interact(local=locals())


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)


