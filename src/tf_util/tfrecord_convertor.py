import collections
from typing import Callable, Dict, Tuple

from data_generator.create_feature import create_int_feature
from tf_util.enum_features import load_record
from tf_util.record_writer_wrap import RecordWriterWrap
from tlm.data_gen.bert_data_gen import create_float_feature


def take(v):
    return v.int64_list.value


def take_float(v):
    return v.float_list.value


def load_record_int64(file_path):
    for record in load_record(file_path):
        d = {}
        for key in record:
            t = take(record[key])
            d[key] = t
        yield d


def extract_convertor(input_tfrecord_path,
                      save_path,
                      convert_fn: Callable[[Dict], Dict]):
    writer = RecordWriterWrap(save_path)
    for record in load_record_int64(input_tfrecord_path):
        d = convert_fn(record)
        od = collections.OrderedDict()
        for key, value in d.items():
            od[key] = create_int_feature(value)
        writer.write_feature(od)


def load_record_int_and_float(file_path):
    for record in load_record(file_path):
        d = {}
        d_float = {}
        for key in record:
            t = take(record[key])
            d[key] = t
            d_float[key] = take_float(record[key])
        yield d, d_float


def extract_convertor_w_float(input_tfrecord_path,
                      save_path,
                      convert_fn: Callable[[Tuple[Dict, Dict]], Tuple[Dict, Dict]]):
    writer = RecordWriterWrap(save_path)
    for record in load_record_int_and_float(input_tfrecord_path):
        d_int, d_float = convert_fn(record)
        od = collections.OrderedDict()
        for key, value in d_int.items():
            od[key] = create_int_feature(value)
        for key, value in d_float.items():
            od[key] = create_float_feature(value)
        writer.write_feature(od)