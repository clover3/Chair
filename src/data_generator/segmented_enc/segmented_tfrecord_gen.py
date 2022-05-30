from typing import List, Dict

from data_generator.segmented_enc.seg_encoder_common import EncoderInterface
from dataset_specific.mnli.mnli_reader import NLIPairData
from tlm.data_gen.bert_data_gen import create_int_feature


def get_encode_fn_from_encoder_list(encoder_list: List[EncoderInterface]):
    def entry_encode(e: NLIPairData) -> Dict:
        text_list = [e.premise, e.hypothesis]
        features = {}
        for i in range(2):
            input_ids, input_mask, segment_ids = encoder_list[i].encode_from_text(text_list[i])
            features["input_ids{}".format(i)] = create_int_feature(input_ids)
            features["input_mask{}".format(i)] = create_int_feature(input_mask)
            features["segment_ids{}".format(i)] = create_int_feature(segment_ids)
        features['label_ids'] = create_int_feature([e.get_label_as_int()])
        return features

    return entry_encode
