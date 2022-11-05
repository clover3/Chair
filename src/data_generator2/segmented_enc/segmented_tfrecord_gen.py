from collections import OrderedDict
from typing import List, Dict

from data_generator2.segmented_enc.seg_encoder_common import SingleEncoderInterface, PairEncoderInterface
from dataset_specific.mnli.mnli_reader import NLIPairData
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