import os
import sys
import numpy as np
from cpath import get_canonical_model_path2, output_path
from data_generator.tokenizer_wo_tf import ids_to_text, get_tokenizer
from misc_lib import two_digit_float
from trainer_v2.custom_loop.dataset_factories import get_sequence_labeling_dataset
from trainer_v2.custom_loop.definitions import ModelConfig600_3
from trainer_v2.custom_loop.run_config2 import get_run_config_for_predict
from trainer_v2.custom_loop.train_loop import load_model_by_dir_or_abs
from trainer_v2.evidence_selector.evidence_candidates import PHSegmentedPairParser
from trainer_v2.train_util.arg_flags import flags_parser


# Q: How many tokens got >0.5 ?


def main(args):
    model_path = get_canonical_model_path2("nli_es1_0", "model_12500")
    run_config = get_run_config_for_predict(args)
    model = load_model_by_dir_or_abs(model_path)
    segment_len = 300
    parser = PHSegmentedPairParser(segment_len)
    eval_files_path = os.path.join(output_path, "align", "evidence_prediction", "train")
    tokenizer = get_tokenizer()
    MASK_ID = tokenizer.wordpiece_tokenizer.vocab["[MASK]"]
    dataset = get_sequence_labeling_dataset(eval_files_path, run_config, ModelConfig600_3(), False)
    for batch in dataset:
        x, y = batch
        output = model.predict_on_batch(x)

        input_ids, segment_ids = x
        for i in range(len(input_ids)):
            pair = parser.get_ph_segment_pair(input_ids[i], segment_ids[i])
            probs = output[i]
            true_probs = probs[:, 0]
            probs_1 = true_probs[:segment_len]
            probs_2 = true_probs[segment_len:]
            probs_i = [probs_1, probs_2]
            print()
            print("p_tokens:", ids_to_text(tokenizer, pair.p_tokens))
            for prem_i in [0, 1]:
                h_tokens = [pair.h1, pair.h2][prem_i]
                print("h{}_tokens: {}".format(prem_i, ids_to_text(tokenizer, h_tokens)))
                p_scores = probs_i[prem_i][1: 1 + len(pair.p_tokens)]
                other_scores = probs_i[prem_i][1 + len(pair.p_tokens):]
                p_tokens = pair.p_tokens
                # p_tokens_del = delete_low80(MASK_ID, p_scores, p_tokens)
                p_tokens_del = list(p_tokens)

                for idx, s in enumerate(p_scores):
                    if s < 0.5:
                        p_tokens_del[idx] = MASK_ID
                print("p{}: {}".format(prem_i, ids_to_text(tokenizer, p_tokens_del)))
                # print(" ".join(map(two_digit_float, p_scores)))



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
