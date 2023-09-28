import logging
import os

from transformers import AutoTokenizer

from cache import save_to_pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
from trainer_v2.chair_logging import c_log, IgnoreFilter, IgnoreFilterRE
import tensorflow as tf

from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2, get_run_config_for_predict_empty
from trainer_v2.custom_loop.train_loop import tf_run2, load_model_by_dir_or_abs
from trainer_v2.train_util.arg_flags import flags_parser

#: Evaluated align prediction accuracy in
#      Contextual setup (holdout)
#      Context-less setup (term enum)

def main(args):
    c_log.info(__file__)
    run_config: RunConfig2 = get_run_config2(args)
    target_q_term = "when"
    query = target_q_term
    dummy_term = "dummy"
    document = dummy_term
    max_seq_length = 256

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    q_term_id = tokenizer.vocab[target_q_term]
    dummy_term_id = tokenizer.vocab[dummy_term]

    def build_dataset():
        encoded_input = tokenizer.encode_plus(
            query,
            document,
            padding="max_length",
            max_length=max_seq_length,
            truncation=True,
        )
        input_ids = encoded_input["input_ids"]
        token_type_ids = encoded_input["token_type_ids"]

        def find_term(term_id, segment_id_val):
            for i in range(len(input_ids)):
                if input_ids[i] == term_id and token_type_ids[i] == segment_id_val:
                    return i
            raise KeyError()

        q_term_idx = find_term(q_term_id, 0)
        doc_term_idx = find_term(dummy_term_id, 1)
        q_term_mask = [0] * len(input_ids)
        q_term_mask[q_term_idx] = 1
        d_term_mask = [0] * len(input_ids)
        d_term_mask[doc_term_idx] = 1

        def gen():
            for i in range(100, 10000):
                yield i
        dataset = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.int32))
        )

        def make_entry(i):
            input_ids_new = list(input_ids)
            input_ids_new[doc_term_idx] = i
            d = {
                "input_ids": input_ids_new,
                "token_type_ids": token_type_ids,
                "q_term_mask": q_term_mask,
                "d_term_mask": d_term_mask,
                "label": [0],
                "is_valid": [1]
            }
            d_pair = {}
            for i in [1, 2]:
                for k, v in d.items():
                    d_pair[f"{k}{i}"] = v
            return d_pair

        dataset = dataset.map(make_entry)
        dataset = dataset.batch(16)
        return dataset

    dataset = build_dataset()
    print(dataset)
    model = load_model_by_dir_or_abs(run_config.eval_config.model_save_path)
    output = model.predict(dataset)
    align_pred = output['align_probe']['all_concat']
    print(align_pred.shape)
    save_to_pickle(output, "galign_pred")


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)



