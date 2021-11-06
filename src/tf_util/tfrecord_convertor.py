import collections
from typing import Callable, Dict

from data_generator.create_feature import create_int_feature
from tf_util.enum_features import load_record
from tf_util.record_writer_wrap import RecordWriterWrap


def take(v):
    return v.int64_list.value


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