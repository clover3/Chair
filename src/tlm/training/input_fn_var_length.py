import os
from collections import OrderedDict

import tensorflow as tf
from tensorflow.python.data import TFRecordDataset

from cpath import output_path
from data_generator.create_feature import create_int_feature
from data_generator.special_tokens import CLS_ID, SEP_ID
from tf_util.record_writer_wrap import RecordWriterWrap
from tlm.training.flags_wrapper import get_input_files_from_flags, show_input_files

feature_names = [
    "input_ids1",
    "input_mask1",
    "segment_ids1",
    "input_ids2",
    "input_mask2",
    "segment_ids2",
]


def concat_vectors(feature):
    map_fn = lambda x: x
    query = map_fn(feature["query"])
    doc1 = map_fn(feature["doc1"])
    doc2 = map_fn(feature["doc2"])
    CLS = [CLS_ID]
    SEP = [SEP_ID]

    def join(query, doc, post_fix):
        input_ids = tf.concat([CLS, query, SEP, doc, SEP], axis=0)
        input_mask = tf.ones_like(input_ids, tf.int32)
        seg0_s = tf.zeros_like(query, tf.int32)
        seg1_s = tf.ones_like(query, tf.int32)
        segment_ids = tf.concat([[0], seg0_s, [0], seg1_s, [1]], axis=0)
        return {
            "input_ids" + post_fix: input_ids,
            "input_mask" + post_fix: input_mask,
            "segment_ids" + post_fix: segment_ids
        }

    s1 = join(query, doc1, "1")
    s2 = join(query, doc2, "2")
    s1.update(s2)
    return tuple([s1[key] for key in feature_names])
    # return s1, s2


def tuple_to_dict(*t):
    return {key: v for key, v in zip(feature_names, t)}


def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.io.parse_single_example(serialized=record, features=name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, dtype=tf.int32)
        example[name] = t

    return example


def get_var_int_ids_spec():
    return tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True)


def build_query_doc12_dataset(input_files, max_seq_length, batch_size) -> TFRecordDataset:
    feature_specs = {
        "query": get_var_int_ids_spec(),
        "doc1": get_var_int_ids_spec(),
        "doc2": get_var_int_ids_spec(),
    }
    d = tf.data.TFRecordDataset(input_files)
    output_buffer_size = batch_size * 1000
    d = d.map(lambda x: _decode_record(x, feature_specs)).prefetch(output_buffer_size)
    d = d.map(concat_vectors)
    d = d.padded_batch(batch_size, tuple([max_seq_length] * 6))
    d = d.map(tuple_to_dict)
    return d


def input_fn_builder(flags,
                     dataset_builder,
                     num_cpu_threads=4,
                     max_pred_steps=0
                     ):
    input_files = get_input_files_from_flags(flags)
    show_input_files(input_files)
    is_training = flags.do_train
    max_seq_length = flags.max_seq_length

    def input_fn(params):
        batch_size = params["batch_size"]
        d: TFRecordDataset = dataset_builder(input_files, batch_size, max_seq_length)
        if is_training:
            cycle_length = min(num_cpu_threads, len(input_files))
            d = d.apply(
                tf.data.experimental.parallel_interleave(
                    tf.data.TFRecordDataset,
                    sloppy=is_training,
                    cycle_length=cycle_length))
            d = d.repeat()
            d = d.shuffle(buffer_size=1000 * 1000)
            cycle_length = min(num_cpu_threads, len(input_files))
            # `sloppy` mode means that the interleaving is not exact. This adds
            d = d.apply(
                tf.data.experimental.parallel_interleave(
                    tf.data.TFRecordDataset,
                    sloppy=is_training,
                    cycle_length=cycle_length))
        else:
            if max_pred_steps > 0:
                d = d.take(max_pred_steps)

        return d

    return input_fn


def save_record():
    writer = RecordWriterWrap(os.path.join(output_path, "q_doc1_doc2.tfrecord"))
    features = {
        "query": create_int_feature([1, 2, 3]),
        "doc1": create_int_feature([1, 2, 3, 4,]),
        "doc2": create_int_feature([1, 2, 3, 4 ,5])
    }
    for _ in range(300):
        writer.write_feature(OrderedDict(features))


def main():
    input_file = os.path.join(output_path, "q_doc1_doc2.tfrecord")
    input_fn = input_fn_builder([input_file], False, 512, build_query_doc12_dataset)
    dataset = input_fn({'batch_size': 11})
    for e in dataset:
        print(e["input_ids1"])
        print(e["input_ids1"].shape)
        break


if __name__ == "__main__":
    main()