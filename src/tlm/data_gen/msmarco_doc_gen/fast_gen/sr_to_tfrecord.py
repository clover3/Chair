import collections

from data_generator.special_tokens import CLS_ID, SEP_ID
from tlm.data_gen.base import get_basic_input_feature_as_list_all_ids
from tlm.data_gen.bert_data_gen import create_int_feature
from tlm.data_gen.msmarco_doc_gen.fast_gen.seg_resource import SegmentRepresentation


def encode_sr(sr: SegmentRepresentation,
              max_seq_length,
              label_id,
              data_id=None) -> collections.OrderedDict:
    s1_len = len(sr.first_seg) + 2
    s2_len = len(sr.second_seg) + 1
    pad_len = max_seq_length - s1_len - s2_len
    segment_ids = s1_len * [0] + s2_len * [1] + pad_len * [0]
    input_mask = (s1_len+s2_len) * [1] + pad_len * [0]
    input_ids = [CLS_ID] + sr.first_seg + [SEP_ID] \
                + sr.second_seg + [SEP_ID] + [0] * pad_len

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    if data_id is not None:
        features["data_id"] = create_int_feature([data_id])
    features["label_ids"] = create_int_feature([label_id])
    return features
