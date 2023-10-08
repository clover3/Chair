import sys
import sys

import numpy as np

from data_generator.tokenizer_wo_tf import get_tokenizer, pretty_tokens
from data_generator2.segmented_enc.es_common.es_two_seg_common import BothSegPartitionedPairParser
from trainer_v2.custom_loop.definitions import ModelConfig512_2
from trainer_v2.custom_loop.run_config2 import get_run_config_for_predict
from trainer_v2.custom_loop.train_loop import load_model_by_dir_or_abs
from trainer_v2.evidence_selector.runner_mmp.dataset_fn import build_state_dataset_fn
from trainer_v2.train_util.arg_flags import flags_parser
import tensorflow as tf


# Q: How many tokens got >0.5 ?


def main(args):
    run_config = get_run_config_for_predict(args)
    model_path = run_config.predict_config.model_save_path
    eval_files_path = run_config.dataset_config.eval_files_path

    model_config = ModelConfig512_2()
    segment_len = int(model_config.max_seq_length / 2)
    model = load_model_by_dir_or_abs(model_path)
    parser = BothSegPartitionedPairParser(segment_len)
    tokenizer = get_tokenizer()
    MASK_ID = tokenizer.wordpiece_tokenizer.vocab["[MASK]"]
    model_config = ModelConfig512_2()
    dataset_builder = build_state_dataset_fn(run_config, model_config)
    dataset = dataset_builder(eval_files_path, False)

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

            pair = parser.parse(input_ids_cur, segment_ids_cur)
            true_pred = output_cur[:, 0]
            pred_1 = true_pred[:segment_len]
            pred_2 = true_pred[segment_len:]
            pred_i = [pred_1, pred_2]
            print()
            evi_tokens = pair.evidence_like_segment.tokens
            print("evi_tokens:", pretty_tokens(evi_tokens, True))
            for part_i in [0, 1]:
                q_like = pair.query_like_segment
                q_like_tokens = [q_like.get_first(), q_like.get_second()][part_i]
                print("query_like {}_tokens: {}".format(part_i, pretty_tokens(q_like_tokens, True)))
                evi_scores = pred_i[part_i][1: 1 + len(evi_tokens)]
                _other_scores = pred_i[part_i][1 + len(evi_tokens):]
                evi_tokens_del = delete_low80("[MASK]", evi_scores, evi_tokens)
                print("evi{}: {}".format(part_i, pretty_tokens(evi_tokens_del, True)))

        break
    pass


def delete_low80(MASK_ID, p_scores, p_tokens):
    del_indices = np.argsort(p_scores)
    n_del = int(len(p_scores) * 0.8)
    p_tokens_del = list(p_tokens)
    for i in del_indices[:n_del]:
        p_tokens_del[i] = MASK_ID
    return p_tokens_del


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
