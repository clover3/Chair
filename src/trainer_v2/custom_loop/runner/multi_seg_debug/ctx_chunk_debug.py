
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
from trainer_v2.custom_loop.modeling_common.tf_helper import distribute_dataset
from trainer_v2.custom_loop.neural_network_def.siamese import ModelConfig2SegProject
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2_nli
from trainer_v2.custom_loop.train_loop_helper import get_strategy_from_config
from trainer_v2.train_util.arg_flags import flags_parser


def save_layer_cache(layer_no, layer_val):
    save_to_pickle(layer_val, "layer_{}_debug_var".format(layer_no))


def get_model_init(model_config):
    inner = CtxChunkInteraction(FuzzyLogicLayerM)
    bert_params = load_bert_config(get_bert_config_path())
    inner.build_model(bert_params, model_config)
    return inner

# Check variables

@report_run3
def main(args):
    c_log.info("Start {}".format(__file__))
    c_log.setLevel(logging.DEBUG)
    run_config: RunConfig2 = get_run_config2_nli(args)
    run_config.common_run_config.batch_size = 2
    model_config = ModelConfig2SegProject()

    def dataset_factory(input_files, is_for_training):
        return get_two_seg_data(input_files, run_config, model_config, is_for_training)

    strategy = get_strategy_from_config(run_config)
    c_log.debug("strategy %s" % strategy)
    with strategy.scope():
        c_log.debug("Building model")
        inner = get_model_init(model_config)

    model = inner.model
    c_log.debug("tf_run_inner initializing dataset")
    eval_dataset = dataset_factory(run_config.dataset_config.eval_files_path, False)
    eval_dataset = distribute_dataset(strategy, eval_dataset)
    debug_vars = inner.get_debug_vars()

    c_log.debug("Model predicts...")
    with strategy.scope():
        new_model = tf.keras.Model(inputs=model.inputs, outputs=debug_vars)
        outputs = new_model.predict(eval_dataset, steps=1)
    c_log.debug("...Done")

    attention_mask = outputs['attention_mask'][0]
    attention_probs = outputs['layer_0_attn_probs'][0]

    p_idx = 2
    h_idx = 200 + 2

    print(f"from={h_idx}, to={h_idx}, maybe 1 actually {attention_mask[h_idx, h_idx]}")
    print(f"from={p_idx}, to={h_idx}, maybe 0 actually {attention_mask[p_idx, h_idx]}")
    print(f"from={h_idx}, to={p_idx}, maybe 1 actually {attention_mask[h_idx, p_idx]}")
    print(f"from={p_idx}, to={p_idx}, maybe 1 actually {attention_mask[p_idx, p_idx]}")
    code.interact(local=locals())

    print(f"from={h_idx}, to={h_idx}, {attention_probs[h_idx, h_idx]} > 0")
    print(f"from={p_idx}, to={h_idx}, {attention_probs[p_idx, h_idx]} == 0")
    print(f"from={h_idx}, to={p_idx}, {attention_probs[h_idx, p_idx]} > 0")
    print(f"from={h_idx}, to={h_idx}, {attention_probs[h_idx, h_idx]} > 0")


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
