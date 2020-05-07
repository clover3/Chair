import collections

from base_type import FilePath
from tf_util.enum_features import load_record_v2
from tf_util.record_writer_wrap import RecordWriterWrap
from tlm.data_gen.bert_data_gen import create_int_feature
from tlm.data_gen.feature_to_text import take


def tfrecord_convertor(source_path: FilePath,
                       output_path: FilePath,
                       feature_transformer
                       ):
    writer = RecordWriterWrap(output_path)
    feature_itr = load_record_v2(source_path)
    for feature in feature_itr:
        new_features = feature_transformer(feature)
        writer.write_feature(new_features)
    writer.close()



def convert_to_2way(source_path, output_path):
    def feature_transformer(feature):
        new_features = collections.OrderedDict()
        mapping = {0: 0,
                   1: 1,
                   2: 1}

        for key in feature:
            v = take(feature[key])
            if key == "label_ids":
                v = [mapping[v[0]]]
            new_features[key] = create_int_feature(v)

        return new_features

    return tfrecord_convertor(source_path, output_path, feature_transformer)
