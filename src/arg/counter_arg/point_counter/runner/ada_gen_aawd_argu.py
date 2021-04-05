import os

from arg.counter_arg.point_counter.ada_gen import combine_source_and_target, get_encode_fn
from arg.counter_arg.point_counter.prepare_data import get_argu_pointwise_data
from cpath import at_output_dir, output_path
from dataset_specific.aawd.load import load_aawd_splits_as_binary
from misc_lib import exist_or_mkdir
from tf_util.record_writer_wrap import write_records_w_encode_fn


def main():
    aawd_train, _, _ = load_aawd_splits_as_binary()
    argu_train = get_argu_point_wise_train_data()

    data_d = {
        'aawd': aawd_train,
        'argu': argu_train
    }

    encode_fn = get_encode_fn(512)
    dir_name = "counter_argument_ada"
    exist_or_mkdir(os.path.join(output_path, dir_name))

    def make_tfrecord(source_name, target_name):
        source_data = data_d[source_name]
        target_data = data_d[target_name]
        combined_data = combine_source_and_target(source_data, target_data, 1)
        save_path = at_output_dir(dir_name, "{}_to_{}_train".format(source_name, target_name))
        write_records_w_encode_fn(save_path, encode_fn, combined_data)

    make_tfrecord("aawd", "argu")
    make_tfrecord("argu", "aawd")


def get_argu_point_wise_train_data():
    train_x, train_y, _, _ = get_argu_pointwise_data()
    argu_train = list(zip(train_x, train_y))
    return argu_train


def train_data_size():
    aawd_train, _, _ = load_aawd_splits_as_binary()
    argu_train = get_argu_point_wise_train_data()

    data_d = {
        'aawd': aawd_train,
        'argu': argu_train
    }

    for data_name, data in data_d.items():
        print(data_name, len(data))


if __name__ == "__main__":
    main()
