
import collections
import random
import sys
from typing import Callable, Dict

from data_generator.create_feature import create_int_feature, create_float_feature
from tf_util.enum_features import load_record
from tf_util.record_writer_wrap import RecordWriterWrap


def take_int(v):
    return v.int64_list.value


def take_float(v):
    return v.float_list.value


def load_record_w_feature_list(file_path, int_features, float_features):
    for record in load_record(file_path):
        d = {}
        for key in int_features:
            t = take_int(record[key])
            d[key] = t
        for key in float_features:
            t = take_float(record[key])
            d[key] = t
        yield d


def extract_selector(input_tfrecord_path,
                     save_path,
                     int_features,
                     float_features,
                     select_fn: Callable[[Dict], Dict]):
    writer = RecordWriterWrap(save_path)
    for record in load_record_w_feature_list(input_tfrecord_path,
                                             int_features, float_features):
        if select_fn(record):
            od = collections.OrderedDict()
            for key in int_features:
                od[key] = create_int_feature(record[key])
            for key in float_features:
                od[key] = create_float_feature(record[key])
            writer.write_feature(od)


def random_true(prob):
    return random.random() < prob


def main():
    def select_fn(record):
        if record['label_ids'][0] > 5:
            return random_true(0.02)
        else:
            return random_true(0.005)

    int_features = ['qtype_id', 'd_e_input_ids', 'd_e_segment_ids',
                 'data_id']
    float_features = ['label_ids']

    extract_selector(sys.argv[1], sys.argv[2], int_features, float_features, select_fn)


if __name__ == "__main__":
    main()