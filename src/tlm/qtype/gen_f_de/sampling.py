import collections
from typing import Callable, Dict, Tuple

from data_generator.create_feature import create_int_feature
from tf_util.record_writer_wrap import RecordWriterWrap
from tf_util.tfrecord_convertor import load_record_int_and_float
from tlm.data_gen.bert_data_gen import create_float_feature


def keep_all_label_ids_as_float(d_pair):
    d_int, d_float = d_pair
    out_d_int = {}
    for key in d_int:
        if not key == "label_ids":
            out_d_int[key] = d_int[key]

    out_d_float = {}
    out_d_float["label_ids"] = d_float["label_ids"]
    return out_d_int, out_d_float


def extract_sampler_w_float(input_tfrecord_path,
                              save_path,
                              convert_fn: Callable[[Tuple[Dict, Dict]], Tuple[Dict, Dict]],
                              sample_rate,
                              ):
    writer = RecordWriterWrap(save_path)
    for idx, record in enumerate(load_record_int_and_float(input_tfrecord_path)):
        if idx % sample_rate == 0:
            pass
        else:
            continue
        d_int, d_float = convert_fn(record)
        od = collections.OrderedDict()
        for key, value in d_int.items():
            od[key] = create_int_feature(value)
        for key, value in d_float.items():
            od[key] = create_float_feature(value)
        writer.write_feature(od)