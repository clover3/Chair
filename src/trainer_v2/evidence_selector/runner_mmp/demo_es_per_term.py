import sys
import sys

import numpy as np

from data_generator.tokenizer_wo_tf import get_tokenizer, pretty_tokens, ids_to_text
from data_generator2.segmented_enc.es_common.partitioned_encoder import BothSegPartitionedPairParser
from trainer_v2.custom_loop.definitions import ModelConfig512_2
from trainer_v2.custom_loop.run_config2 import get_run_config_for_predict
from trainer_v2.custom_loop.train_loop import load_model_by_dir_or_abs
from trainer_v2.evidence_selector.environment_qd import ConcatMaskStrategyQD
from trainer_v2.evidence_selector.runner_mmp.dataset_fn import build_state_dataset_fn
from trainer_v2.train_util.arg_flags import flags_parser
import tensorflow as tf
from typing import List, Iterable, Callable, Dict, Tuple, Set


# Q: How many tokens got >0.5 ?



def select_top_k(p_scores, p_tokens, k) -> List[str]:
    top_k = np.argsort(p_scores)[::-1][:k]
    output = []
    for i, token in enumerate(p_tokens):
        if i in top_k:
            output.append(token)
    return output


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

    masker = ConcatMaskStrategyQD()
    cnt = 0
    for batch in dataset:
        x, y = batch
        output = model.predict_on_batch(x)

        k = 5
        input_ids, segment_ids = x
        n_inst = len(input_ids)
        for pair_i in range(n_inst):
            input_ids_cur = input_ids[pair_i].numpy()
            segment_ids_cur = segment_ids[pair_i].numpy()
            output_cur = output[pair_i][:, 1]

            # get query tokens
            q_tokens_ids = []
            for i_id, s_id in zip(input_ids_cur, segment_ids_cur):
                if s_id == 0 and i_id != 0:
                    q_tokens_ids.append(i_id)

            d_tokens_ids = []
            for i_id, s_id in zip(input_ids_cur, segment_ids_cur):
                if s_id == 1 and i_id != 0:
                    d_tokens_ids.append(i_id)

            print("query_like tokens: {}".format(ids_to_text(tokenizer, q_tokens_ids)))
            print("evidence tokens: {}".format(ids_to_text(tokenizer, d_tokens_ids)))

            bias = masker.get_deletable_evidence_mask(input_ids_cur, segment_ids_cur)
            output_cur_masked = output_cur - bias * (-99999.)
            top_k_indices = np.argsort(output_cur_masked)[::-1][:k]

            sel_tokens = []
            for i, token in enumerate(input_ids_cur):
                if i in top_k_indices:
                    sel_tokens.append(token)
            print("evi: {}".format(ids_to_text(tokenizer, sel_tokens)))

        cnt += 1
        if cnt > 10:
            break

if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
