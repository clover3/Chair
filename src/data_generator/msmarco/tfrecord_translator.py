from functools import partial

from misc_lib import slice_iterator
from tf_util.enum_features import load_record
from tlm.data_gen.feature_to_text import take


def transform(max_seq_length, feature):
    query_ids = take(feature["query_ids"])
    doc_ids = feature["doc_ids"].int64_list.value
    label_ids = feature["label"].int64_list.value[0]

    input_ids = list(query_ids)+ list(doc_ids)
    segment_ids = [0] * len(query_ids) + [1] * len(doc_ids)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    return input_ids, input_mask, segment_ids, label_ids


def translate(tfrecord_path, st, ed):
    max_seq_length = 512
    transform_fn = partial(transform, max_seq_length)
    itr = slice_iterator(load_record(tfrecord_path), st, ed)
    for entry in itr:
        yield transform_fn(entry)


