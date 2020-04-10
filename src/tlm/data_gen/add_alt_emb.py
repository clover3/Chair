import collections
from typing import List, Set

from base_type import FilePath
from list_lib import flatten
from tf_util.enum_features import load_record_v2
from tf_util.record_writer_wrap import RecordWriterWrap
from tlm.data_gen.bert_data_gen import create_int_feature
from tlm.data_gen.feature_to_text import take


def tfrecord_convertor_with_none(source_path: FilePath,
                       output_path: FilePath,
                       feature_transformer
                       ):
    writer = RecordWriterWrap(output_path)
    feature_itr = load_record_v2(source_path)
    for feature in feature_itr:
        new_features = feature_transformer(feature)
        if new_features is not None:
            writer.write_feature(new_features)

    writer.close()


def convert_alt_emb(source_path, output_path, seq_set: List[List[int]]):
    all_tokens: Set[int] = set(flatten(seq_set))
    min_overlap = min([len(set(tokens)) for tokens in seq_set])

    def feature_transformer(feature):
        new_features = collections.OrderedDict()
        success = False
        for key in feature:
            v = take(feature[key])
            if key == "input_ids":
                alt_emb_mask = [0] * len(v)
                s = set(v)
                if len(s.intersection(all_tokens)) >= min_overlap:
                    for word in seq_set:
                        pre_match = 0
                        for i in range(len(v)):
                            if v[i] == word[pre_match]:
                                pre_match += 1
                            else:
                                pre_match = 0
                            if pre_match == len(word):
                                pre_match = 0
                                for j in range(i - len(word) + 1, i+1):
                                    alt_emb_mask[j] = 1
                                    success = True
                new_features["alt_emb_mask"] = create_int_feature(alt_emb_mask)
            new_features[key] = create_int_feature(v)

        if success:
            return new_features
        else:
            return None

    return tfrecord_convertor_with_none(source_path, output_path, feature_transformer)
