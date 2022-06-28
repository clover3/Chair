
import code
import logging
import os

from trainer_v2.custom_loop.neural_network_def.multi_segments import ChunkStartEncoder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from cpath import get_bert_config_path
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config
from trainer_v2.custom_loop.neural_network_def.segmented_enc import FuzzyLogicLayerM2

import sys
import tensorflow as tf

from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.dataset_factories import get_two_seg_data
from trainer_v2.custom_loop.modeling_common.tf_helper import distribute_dataset
from trainer_v2.custom_loop.neural_network_def.siamese import ModelConfig2SegProject
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2_nli
from trainer_v2.custom_loop.train_loop_helper import get_strategy_from_config
from trainer_v2.train_util.arg_flags import flags_parser



def get_model_init(model_config):
    inner = ChunkStartEncoder(FuzzyLogicLayerM2)
    bert_params = load_bert_config(get_bert_config_path())
    inner.build_model(bert_params, model_config)
    return inner

# Check variables

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
    # attention_probs = outputs['layer_0_attn_probs'][0]

    src_idx = 2
    trg_idx = 7

    for i in range(30):
        print(attention_mask[i][:30].tolist())

    print(f"from={trg_idx}, to={trg_idx}, maybe 1 actually {attention_mask[trg_idx, trg_idx]}")
    print(f"from={src_idx}, to={trg_idx}, maybe 1 actually {attention_mask[src_idx, trg_idx]}")
    print(f"from={trg_idx}, to={src_idx}, maybe 1 actually {attention_mask[trg_idx, src_idx]}")
    print(f"from={src_idx}, to={src_idx}, maybe 1 actually {attention_mask[src_idx, src_idx]}")
    code.interact(local=locals())
    #
    # print(f"from={trg_idx}, to={trg_idx}, {attention_probs[trg_idx, trg_idx]} > 0")
    # print(f"from={src_idx}, to={trg_idx}, {attention_probs[src_idx, trg_idx]} > 0")
    # print(f"from={trg_idx}, to={src_idx}, {attention_probs[trg_idx, src_idx]} > 0")
    # print(f"from={trg_idx}, to={trg_idx}, {attention_probs[trg_idx, trg_idx]} > 0")


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
