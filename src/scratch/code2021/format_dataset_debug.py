import os
from collections import OrderedDict

import tensorflow as tf
from tensorflow.python.ops.parsing_ops import FixedLenFeature

from cpath import output_path
from misc_lib import exist_or_mkdir
from tf_util.record_writer_wrap import write_records_w_encode_fn
from tlm.data_gen.bert_data_gen import create_int_feature
from tlm.training.input_fn_common import _decode_record


def make_datasets():
    def encode(number: int) -> OrderedDict:
        feature = OrderedDict()
        feature['value'] = create_int_feature([number])
        return feature

    save_dir = os.path.join(output_path, "format_dataset_debug")
    exist_or_mkdir(save_dir)
    for file_no in range(200):
        save_path = os.path.join(save_dir, str(file_no))
        data = [file_no] * 1000
        write_records_w_encode_fn(save_path, encode, data)

def get_input_files():
    save_dir = os.path.join(output_path, "format_dataset_debug")
    exist_or_mkdir(save_dir)
    for file_no in range(200):
        save_path = os.path.join(save_dir, str(file_no))
        yield save_path


def run():
    input_files = list(get_input_files())
    d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
    d = d.shuffle(buffer_size=10)
    for file in d:
        print(file)

    # `cycle_length` is the number of parallel files that get read.
    cycle_length = 50
    # `sloppy` mode means that the interleaving is not exact. This adds
    # even more randomness to the training pipeline.
    block_length = 4
    d = d.apply(
        tf.data.experimental.parallel_interleave(
            tf.data.TFRecordDataset,
            sloppy=True,
            block_length=block_length,
            cycle_length=cycle_length))
    d = d.shuffle(buffer_size=100)
    features = {
        "value": FixedLenFeature([1], tf.int64),
    }
    d = d.apply(
        tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, features),
            batch_size=4,
            num_parallel_batches=4,
            drop_remainder=True))

    for step, item in enumerate(d):
        print(step, item['value'].numpy())


if __name__ == "__main__":
    run()