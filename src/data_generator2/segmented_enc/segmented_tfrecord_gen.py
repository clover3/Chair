import os
from collections import OrderedDict
from typing import List, Dict

from cpath import output_path
from data_generator2.segmented_enc.seg_encoder_common import SingleEncoderInterface, PairEncoderInterface
from dataset_specific.mnli.mnli_reader import NLIPairData
from misc_lib import path_join, exist_or_mkdir
from tf_util.record_writer_wrap import write_records_w_encode_fn
from tlm.data_gen.bert_data_gen import create_int_feature


def get_encode_fn_from_encoder_list(encoder_list: List[SingleEncoderInterface]):
    def entry_encode(e: NLIPairData) -> Dict:
        text_list = [e.premise, e.hypothesis]
        features = OrderedDict()
        for i in range(2):
            input_ids, input_mask, segment_ids = encoder_list[i].encode_from_text(text_list[i])
            features["input_ids{}".format(i)] = create_int_feature(input_ids)
            features["input_mask{}".format(i)] = create_int_feature(input_mask)
            features["segment_ids{}".format(i)] = create_int_feature(segment_ids)
        features['label_ids'] = create_int_feature([e.get_label_as_int()])
        return features

    return entry_encode


def get_encode_fn_from_encoder(encoder: PairEncoderInterface):
    def entry_encode(e: NLIPairData) -> Dict:
        features = OrderedDict()
        input_ids, input_mask, segment_ids = encoder.encode_from_text(e.premise, e.hypothesis)
        features["input_ids"] = create_int_feature(input_ids)
        features["input_mask"] = create_int_feature(input_mask)
        features["segment_ids"] = create_int_feature(segment_ids)
        features['label_ids'] = create_int_feature([e.get_label_as_int()])
        return features

    return entry_encode


def encode_triplet(triplet, label: int):
    features = OrderedDict()
    input_ids, input_mask, segment_ids = triplet
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features['label_ids'] = create_int_feature([label])
    return features


def encode_seq_prediction(input_ids, input_mask, segment_ids, label_ids):
    features = OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features['label_ids'] = create_int_feature(label_ids)
    return features


def gen_concat_two_seg(reader, encoder, data_name, split):
    output_dir = path_join(output_path, "tfrecord", data_name)
    exist_or_mkdir(output_dir)
    save_path = os.path.join(output_dir, split)
    encode_fn = get_encode_fn_from_encoder(encoder)
    write_records_w_encode_fn(save_path, encode_fn, reader.load_split(split), reader.get_data_size(split))