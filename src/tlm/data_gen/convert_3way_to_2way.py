import collections

from tf_util.enum_features import load_record_v2
from tf_util.record_writer_wrap import RecordWriterWrap
from tlm.data_gen.bert_data_gen import create_int_feature
from tlm.data_gen.feature_to_text import take


def convert(source_path, output_path):
    writer = RecordWriterWrap(output_path)
    feature_itr = load_record_v2(source_path)
    mapping = {0: 0,
               1: 1,
               2: 1}
    for feature in feature_itr:
        new_features = collections.OrderedDict()

        for key in feature:
            v = take(feature[key])
            if key == "label_ids":
                v = [mapping[v[0]]]
            new_features[key] = create_int_feature(v)

        writer.write_feature(new_features)
    writer.close()
