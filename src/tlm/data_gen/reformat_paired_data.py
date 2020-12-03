
import collections

from tf_util.enum_features import load_record_v2
from tf_util.record_writer_wrap import RecordWriterWrap
from tlm.data_gen.bert_data_gen import create_int_feature
from tlm.data_gen.feature_to_text import take


def convert_to_unpaired(source_path, output_path):
    def feature_transformer(feature):
        new_features_1 = collections.OrderedDict()
        new_features_2 = collections.OrderedDict()

        def put(feature_name):
            return create_int_feature(take(feature[feature_name]))

        new_features_1["input_ids"] = put("input_ids1")
        new_features_1["input_mask"] = put("input_mask1")
        new_features_1["segment_ids"] = put("segment_ids1")
        new_features_1["label_ids"] = create_int_feature([1])

        new_features_2["input_ids"] = put("input_ids2")
        new_features_2["input_mask"] = put("input_mask2")
        new_features_2["segment_ids"] = put("segment_ids2")
        new_features_2["label_ids"] = create_int_feature([0])

        return new_features_1, new_features_2

    writer = RecordWriterWrap(output_path)
    feature_itr = load_record_v2(source_path)
    for feature in feature_itr:
        new_features_1, new_features_2 = feature_transformer(feature)
        writer.write_feature(new_features_1)
        writer.write_feature(new_features_2)
    writer.close()

