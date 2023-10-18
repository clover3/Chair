import os.path
import pickle
import sys
import sys

import numpy as np

from data_generator.tokenizer_wo_tf import get_tokenizer, pretty_tokens
from data_generator2.segmented_enc.es_common.es_two_seg_common import EvidencePair2
from data_generator2.segmented_enc.es_common.partitioned_encoder import BothSegPartitionedPairParser
from misc_lib import TimeEstimator
from trainer_v2.custom_loop.definitions import ModelConfig512_2
from trainer_v2.custom_loop.run_config2 import get_run_config_for_predict
from trainer_v2.custom_loop.train_loop import load_model_by_dir_or_abs
from trainer_v2.evidence_selector.runner_mmp.dataset_fn import build_state_dataset_fn
from trainer_v2.train_util.arg_flags import flags_parser
import tensorflow as tf



def main(args):
    run_config = get_run_config_for_predict(args)
    model_path = run_config.predict_config.model_save_path
    model = load_model_by_dir_or_abs(model_path)

    tfrecord_path = run_config.dataset_config.eval_files_path

    model_config = ModelConfig512_2()
    segment_len = int(model_config.max_seq_length / 2)
    parser = BothSegPartitionedPairParser(segment_len)
    model_config = ModelConfig512_2()
    dataset_builder = build_state_dataset_fn(run_config, model_config)
    dataset = dataset_builder(tfrecord_path, False)

    data_size = 30000
    ticker = TimeEstimator(data_size, sample_size=100)

    output_rows = []
    for batch in dataset:
        x, y = batch
        output = model.predict_on_batch(x)
        input_ids, segment_ids = x
        n_inst = int(len(input_ids) / 2)
        for pair_i in range(n_inst):
            pair_i_base = pair_i * 2
            input_ids_cur = tf.concat([input_ids[pair_i_base], input_ids[pair_i_base+1]], axis=0)
            segment_ids_cur = tf.concat([segment_ids[pair_i_base], segment_ids[pair_i_base+1]], axis=0)
            output_cur = tf.concat([output[pair_i_base], output[pair_i_base+1]], axis=0)
            pair: EvidencePair2 = parser.parse(input_ids_cur, segment_ids_cur)
            row = pair, output_cur.numpy()
            output_rows.append(row)
            ticker.tick()

    pickle.dump(output_rows, open(run_config.predict_config.predict_save_path, "wb"))


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
