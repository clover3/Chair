import collections

from tlm.data_gen.bert_data_gen import create_int_feature
from tlm.data_gen.convert_3way_to_2way import tfrecord_convertor
from tlm.data_gen.feature_to_text import take


def tfrecord_segment_ids_convert(source_path, output_path):
    def feature_transformer(feature):
        new_features = collections.OrderedDict()
        mapping = {0: 0,
                   1: 0,
                   2: 1}

        for key in feature:
            l = take(feature[key])
            if key == "segment_ids":
                l = list([mapping[v] for v in l])

            new_features[key] = create_int_feature(l)

        return new_features

    return tfrecord_convertor(source_path, output_path, feature_transformer)

