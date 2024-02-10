from collections import OrderedDict

from data_generator.create_feature import create_int_feature


def combine_with_sep_cls_old(tokenizer, tokens1, tokens2, max_seq_length):
    # I don't know why but this function is slow.
    encoded_input = tokenizer.encode_plus(
        tokens1,
        tokens2,
        padding="max_length",
        max_length=max_seq_length,
        truncation=True,
        is_split_into_words=True,
        # return_tensors="tf"
    )
    return encoded_input['input_ids'][0], encoded_input['token_type_ids'][0]



def combine_with_sep_cls_inner(max_seq_length, tokens1, tokens2):
    max_seg2_len = max_seq_length - 3 - len(tokens1)
    if len(tokens2) > max_seg2_len:
        print("Cut {} to {}".format(len(tokens2), max_seg2_len))
    tokens2 = tokens2[:max_seg2_len]
    tokens = ["[CLS]"] + tokens1 + ["[SEP]"] + tokens2 + ["[SEP]"]
    segment_ids = [0] * (len(tokens1) + 2) \
                  + [1] * (len(tokens2) + 1)

    tokens = tokens[:max_seq_length]
    segment_ids = segment_ids[:max_seq_length]
    return tokens, segment_ids


def combine_with_sep_cls_and_pad(tokenizer, tokens1, tokens2, max_seq_length):
    tokens, segment_ids = combine_with_sep_cls_inner(max_seq_length, tokens1, tokens2)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    return get_basic_input_feature_as_list_all_ids(input_ids, segment_ids, max_seq_length)


def get_basic_input_feature_as_list_all_ids(input_ids, segment_ids, max_seq_length):
    segment_ids = list(segment_ids)
    max_seq_length = max_seq_length
    assert len(input_ids) <= max_seq_length
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        segment_ids.append(0)
    assert len(input_ids) == max_seq_length
    assert len(segment_ids) == max_seq_length
    return input_ids, segment_ids


def encode_pair(pair, label: int):
    features = OrderedDict()
    input_ids, segment_ids = pair
    features["input_ids"] = create_int_feature(input_ids)
    features["segment_ids"] = create_int_feature(segment_ids)
    features['label_ids'] = create_int_feature([label])
    return features


def encode_seg_pair_paired(pair1, pair2):
    features = OrderedDict()
    input_ids1, segment_ids1 = pair1
    features["input_ids1"] = create_int_feature(input_ids1)
    features["segment_ids1"] = create_int_feature(segment_ids1)

    input_ids2, segment_ids2 = pair2
    features["input_ids2"] = create_int_feature(input_ids2)
    features["segment_ids2"] = create_int_feature(segment_ids2)
    return features