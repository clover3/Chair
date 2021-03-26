import os

from arg.counter_arg.point_counter.svm_experiment import get_data
from arg.counter_arg.point_counter.tf_encoder import get_encode_fn
from cpath import at_output_dir, output_path
from misc_lib import exist_or_mkdir
from tf_util.record_writer_wrap import write_records_w_encode_fn


def binary_gen():
    exist_or_mkdir(os.path.join(output_path, "argu_ana_tfrecord"))
    train_x, train_y, dev_x, dev_y = get_data()
    train = zip(train_x, train_y)
    dev = zip(dev_x, dev_y)
    todo = [(train, "train"), (dev, "dev")]
    encode_fn = get_encode_fn(512)
    for data, split in todo:
        save_path = at_output_dir("argu_ana_tfrecord", split)
        write_records_w_encode_fn(save_path, encode_fn, data)


if __name__ == "__main__":
    binary_gen()
