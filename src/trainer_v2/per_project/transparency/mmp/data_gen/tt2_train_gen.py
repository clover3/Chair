from collections import OrderedDict
from typing import Tuple

from tlm.data_gen.bert_data_gen import create_int_feature, create_float_feature
from trainer_v2.per_project.transparency.transformers_utils import pad_truncate


def get_encode_fn_when_bow(process_fn):
    def encode_fn(q_pos_neg: Tuple[str, str, str]):
        q, d1, d2 = q_pos_neg
        feature: OrderedDict = OrderedDict()
        def encode_qd(q, d, i):
            score_, feature_ids_, feature_values_ = process_fn(q, d)
            feature_ids_ = pad_truncate(feature_ids_, 32)
            feature_values_ = pad_truncate(feature_values_, 32)

            feature[f"feature_ids{i}"] = create_int_feature(feature_ids_)
            feature[f"feature_values{i}"] = create_int_feature(feature_values_)
            feature[f"score{i}"] = create_float_feature([score_])
        encode_qd(q, d1, 1)
        encode_qd(q, d2, 2)
        return feature

    return encode_fn