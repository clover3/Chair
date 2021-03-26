import os

from arg.counter_arg.point_counter.tf_encoder import get_encode_fn
from cpath import at_output_dir, output_path
from dataset_specific.aawd.load import load_aawd_splits, load_aawd_splits_as_binary
from misc_lib import exist_or_mkdir
from tf_util.record_writer_wrap import write_records_w_encode_fn


def main():
    exist_or_mkdir(os.path.join(output_path, "aawd_tfrecord"))
    train, dev, test = load_aawd_splits()
    todo = [(train, "train"), (dev, "dev"), (test, "test")]
    encode_fn = get_encode_fn(256)
    for data, split in todo:
        save_path = at_output_dir("aawd_tfrecord", split)
        write_records_w_encode_fn(save_path, encode_fn, data)


def binary_gen():
    exist_or_mkdir(os.path.join(output_path, "aawd_tfrecord_binary"))
    train, dev, test = load_aawd_splits_as_binary()
    todo = [(train, "train"), (dev, "dev"), (test, "test")]
    encode_fn = get_encode_fn(512)
    for data, split in todo:
        save_path = at_output_dir("aawd_tfrecord_binary", split)
        write_records_w_encode_fn(save_path, encode_fn, data)


if __name__ == "__main__":
    binary_gen()
