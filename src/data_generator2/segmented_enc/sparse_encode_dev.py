import os
from cpath import output_path
from misc_lib import path_join

from collections import OrderedDict
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.parsing_ops import SparseFeature

from tf_util.record_writer_wrap import write_records_w_encode_fn
from tlm.data_gen.bert_data_gen import create_int_feature, create_float_feature


def encode_example(pair) -> OrderedDict:
    input_ids, dense_vector = pair
    features = OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    sparse_tensor = tf.sparse.from_dense(dense_vector)
    features["sparse_tensor_indices"] = create_int_feature(tf.reshape(sparse_tensor.indices, [-1]))
    features["sparse_tensor_values"] = create_float_feature(sparse_tensor.values)
    return features


def write():
    def iterate_dummy_data():
        for i in range(10):
            input_ids = [1] * (i+1)
            sparse_feature = np.zeros([30222])
            k_list = []
            for k in [1010, 1012, 1024, 1029, 1037, 1996, 2003, 2054, 2101,
                      3426, 3684, 4079, 4658, 5072, 5646, 6911, 9829, 13233]:
                if (i+k) % 10 == 0:
                    k_list.append(k)
            for k in k_list:
                sparse_feature[k] = 1
            print(input_ids, k_list)
            yield input_ids, sparse_feature

    save_path = path_join(output_path, "sparse_dev.tfrecord")
    write_records_w_encode_fn(save_path, encode_example, iterate_dummy_data())


def read():
    record_path = path_join(output_path, "sparse_dev.tfrecord")
    name_to_features = {
        'input_ids': tf.io.RaggedFeature(tf.int64),
        'sparse_tensor_indices': tf.io.RaggedFeature(tf.int64),
        'sparse_tensor_values': tf.io.RaggedFeature(tf.float32),
    }

    vector_len = 30222

    def decode_record(record):
        record = tf.io.parse_single_example(record, name_to_features)
        tensor = tf.sparse.SparseTensor(
            indices=tf.expand_dims(record['sparse_tensor_indices'], axis=1),
            values=record['sparse_tensor_values'],
            dense_shape=[vector_len, ]
        )
        return {
            'input_ids': record['input_ids'],
            'tensor': tf.sparse.to_dense(tensor),
        }
    dataset = tf.data.TFRecordDataset([record_path])
    # raw_example = next(iter(dataset))
    # parsed = tf.train.Example.FromString(raw_example.numpy())

    dataset = dataset.map(decode_record,
                          num_parallel_calls=tf.data.AUTOTUNE)
    print(dataset)

    for item in dataset:
        print(item)


if __name__ == "__main__":
    write()
    read()