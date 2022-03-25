import sys
from typing import List, Dict

import os
import sys
from typing import List, Dict

import numpy.random
import tensorflow as tf
from tensorflow.python.data import Dataset
from transformers import MPNetTokenizer

from cache import save_list_to_jsonl, load_list_from_jsonl
from cpath import output_path
from misc_lib import TimeEstimator
from trainer_v2.arg_flags import flags_parser
from trainer_v2.epr.input_fn import get_dataset_fn
from trainer_v2.epr.mpnet import TFSBERT
from trainer_v2.epr.path_helper import get_segmented_data_path
from trainer_v2.get_tpu_strategy import get_strategy


def align(p_vectors, h_vectors):
    norm_p = tf.nn.l2_normalize(p_vectors, -1)
    norm_h = tf.nn.l2_normalize(h_vectors, -1)
    score_matrix = tf.matmul(norm_p, norm_h, transpose_b=True)
    max_p_idx_for_h = tf.argmax(score_matrix, axis=1)
    max_h_idx_for_p = tf.argmax(score_matrix, axis=2)
    return max_p_idx_for_h, max_h_idx_for_p



def load_segmented_data(dataset_name, split) -> List[Dict]:
    file_path = get_segmented_data_path(dataset_name, split)
    return load_list_from_jsonl(file_path, lambda x: x)


def load_dataset_from_json(feature_keys, segment_keys, segment_per_text):
    json_iter = load_segmented_data("snli", "validation")
    tokenizer = MPNetTokenizer.from_pretrained("microsoft/mpnet-base")
    def generator(json_iter):
        def pad_items(item):
            if len(item) > segment_per_text:
                return item[:segment_per_text]
            else:
                pad_len = segment_per_text - len(item)
                padded = item + [""] * pad_len
                return padded

        def convert_text_list(text_list):
            input_ids = tokenizer(pad_items(text_list))
            return input_ids

        for e in json_iter:
            d = {
                'label': e['label']
            }
            for segment in segment_keys:
                per_seg = convert_text_list(e[segment])
                for feature_name in feature_keys:
                    new_name = f"{segment}_{feature_name}"
                    d[new_name] = tf.ragged.constant(per_seg[feature_name])
            yield d

    feature_spec = tf.RaggedTensorSpec(shape=(segment_per_text, None), dtype=tf.int32)
    spec = {
        'label': tf.TensorSpec(shape=(), dtype=tf.int32),
    }
    spec = (tf.TensorSpec(shape=(), dtype=tf.int32),)
    for segment in segment_keys:
        for feature_name in feature_keys:
            new_name = f"{segment}_{feature_name}"
            # spec.append(feature_spec)
            # spec[new_name] = feature_spec
    print(spec)
    # dataset = Dataset.from_generator(lambda: generator(json_iter),
    # output_signature=spec)

    def gen():
        ragged_tensor = tf.ragged.constant([[1, 2], [3]])
        yield 42, ragged_tensor
    dataset = Dataset.from_generator(gen,
                                     output_signature=(
                                         tf.TensorSpec(shape=(), dtype=tf.int32),
                                         tf.RaggedTensorSpec(shape=(2, None), dtype=tf.int32))
                                     )

    return dataset


def load_dataset_dummy():
    json_iter = load_segmented_data("snli", "validation")
    tokenizer = MPNetTokenizer.from_pretrained("microsoft/mpnet-base")
    def generator(json_iter):
        for e in range(100):
            d = {
                'label': 1,
                'inputs': numpy.ones([16])
            }
            yield d

    spec = {
        'label': tf.TensorSpec(shape=(), dtype=tf.int32),
        'inputs_ids': tf.TensorSpec(shape=(16, 4), dtype=tf.int32),
        'attention_mask': tf.TensorSpec(shape=(16, 4), dtype=tf.int32),
    }
    dataset = Dataset.from_generator(lambda: generator(json_iter), output_signature=spec)
    return dataset


def run_inner(model_path, config_path, input_files):
    segment_per_text = 16
    batch_size = 4

    feature_keys = ["input_ids", "attention_mask"]
    segment_keys = ["premise", "hypothesis"]

    # dataset = load_dataset_from_json(feature_keys, segment_keys, segment_per_text)
    dataset = get_dataset_fn(input_files, batch_size, False)()
    # dataset = dataset.batch(batch_size)
    sbert = TFSBERT(model_path, config_path)
    out_e_list = []
    json_iter = load_segmented_data("snli", "validation")
    ticker = TimeEstimator(len(json_iter))
    print("Start iteration")
    for e in dataset:
        def get_avg_vector(segment):
            feature_per_segment = {}
            for feature_name in feature_keys:
                new_name = f"{segment}_{feature_name}"
                tensor = tf.reshape(e[new_name], [batch_size * segment_per_text, -1])
                feature_per_segment[feature_name] = tensor
            avg_vectors = sbert.predict_from_ids(feature_per_segment)
            return tf.reshape(avg_vectors, [batch_size, segment_per_text, -1])

        p_avg_vectors = get_avg_vector('premise')
        h_avg_vectors = get_avg_vector('hypothesis')
        max_p_idx_for_h, max_h_idx_for_p = align(p_avg_vectors, h_avg_vectors)
        e['max_p_idx_for_h'] = max_p_idx_for_h
        e['max_h_idx_for_p'] = max_h_idx_for_p
        ticker.tick()
        out_e_list.append(e)

    print("End iteration")
    for e in out_e_list:
        for key in e:
            e[key] = e[key].numpy().tolist()
    save_path = os.path.join(output_path, "temp")
    save_list_to_jsonl(out_e_list, save_path)


def main(arg):
    print("args.use_tpu", args.use_tpu)
    strategy = get_strategy(args.use_tpu, args.tpu_name)
    with strategy.scope():
        run_inner(arg.init_checkpoint, arg.config_path, arg.input_files)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)

