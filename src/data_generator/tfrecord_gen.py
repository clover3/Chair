import collections

from data_generator.create_feature import create_int_feature
from data_generator.tokenizer_wo_tf import get_tokenizer
from tf_util.record_writer_wrap import RecordWriterWrap


def entry_to_feature_dict(e):
    input_ids, input_mask, segment_ids, label = e
    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features["label_ids"] = create_int_feature([label])
    return features


def pairwise_entry_to_feature_dict(pair):
    features = collections.OrderedDict()
    for idx, e in enumerate(pair):
        input_ids, input_mask, segment_ids, label = e
        features["input_ids"+str(idx+1)] = create_int_feature(input_ids)
        features["input_mask"+str(idx+1)] = create_int_feature(input_mask)
        features["segment_ids"+str(idx+1)] = create_int_feature(segment_ids)
        features["label_ids"+str(idx+1)] = create_int_feature([label])
    return features


def modify_data_loader(data_loader):
    tokenizer = get_tokenizer()
    CLS_ID = tokenizer.convert_tokens_to_ids(["[CLS]"])[0]
    SEP_ID = tokenizer.convert_tokens_to_ids(["[SEP]"])[0]
    data_loader.CLS_ID_3 = CLS_ID
    data_loader.SEP_ID_4 = SEP_ID
    return data_loader


def write_features_to_file(data, output_file):
    writer = RecordWriterWrap(output_file)
    for t in data:
        writer.write_feature(t)
    writer.close()
    return writer