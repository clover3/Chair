import collections

from data_generator.bert_input_splitter import split_p_h_with_input_ids
from tf_util.record_writer_wrap import RecordWriterWrap
from tlm.data_gen.bert_data_gen import create_int_feature


def transform_datapoint(data_point):
    input_ids = data_point['input_ids']
    max_seq_length = len(input_ids)
    assert max_seq_length == 200
    p, h = split_p_h_with_input_ids(input_ids, input_ids)
    segment_ids = (2+len(p)) * [0] + (1+len(h)) * [1]
    input_mask = (3+len(p)+len(h)) * [1]

    while len(segment_ids) < max_seq_length:
        input_mask.append(0)
        segment_ids.append(0)
    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features["label_ids"] = create_int_feature([data_point['label']])
    return features


def save_to_tfrecord(new_data, out_path):
    writer = RecordWriterWrap(out_path)
    for dp in new_data:
        writer.write_feature(transform_datapoint(dp))
    writer.close()