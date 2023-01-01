import collections
from typing import List

import tlm.data_gen.bert_data_gen as btd


def pad0(seq, max_len):
    assert len(seq) <= max_len
    while len(seq) < max_len:
        seq.append(0)
    return seq


def truncate_seq(tokens_a, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a)
        if total_length <= max_num_tokens:
            break

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del tokens_a[0]
        else:
            tokens_a.pop()
    return tokens_a


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    cnt = 0
    while True:
        cnt += 1
        if cnt > 912 and cnt % 100 == 0:
            print("Infinited loop :")
            print(tokens_a)
            print(tokens_b)
            print(max_num_tokens)
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


def format_tokens_n_segid(raw_tokens):
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in raw_tokens:
        tokens.append(token)
        segment_ids.append(0)

    tokens.append("[SEP]")
    segment_ids.append(0)
    return tokens, segment_ids


def format_tokens_pair_n_segid(tokens_a, tokens_b):
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)

    tokens.append("[SEP]")
    segment_ids.append(0)

    for token in tokens_b:
        tokens.append(token)
        segment_ids.append(1)

    tokens.append("[SEP]")
    segment_ids.append(1)

    return tokens, segment_ids


def get_basic_input_feature_as_list(tokenizer, max_seq_length, input_tokens, segment_ids):
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    return get_basic_input_feature_as_list_all_ids(input_ids, segment_ids, max_seq_length)


def get_basic_input_feature_as_list_all_ids(input_ids, segment_ids, max_seq_length):
    input_mask = [1] * len(input_ids)
    segment_ids = list(segment_ids)
    max_seq_length = max_seq_length
    assert len(input_ids) <= max_seq_length
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    return input_ids, input_mask, segment_ids


def get_basic_input_feature(tokenizer, max_seq_length, input_tokens, segment_ids):
    input_ids, input_mask, segment_ids = get_basic_input_feature_as_list(tokenizer, max_seq_length, input_tokens, segment_ids)
    return ordered_dict_from_input_segment_mask_ids(input_ids, input_mask, segment_ids)


def ordered_dict_from_input_segment_mask_ids(input_ids, input_mask, segment_ids):
    features = collections.OrderedDict()
    features["input_ids"] = btd.create_int_feature(input_ids)
    features["input_mask"] = btd.create_int_feature(input_mask)
    features["segment_ids"] = btd.create_int_feature(segment_ids)
    return features


def combine_with_sep_cls(max_seq_length, tokens1, tokens2):
    max_seg2_len = max_seq_length - 3 - len(tokens1)
    tokens2 = tokens2[:max_seg2_len]
    tokens = ["[CLS]"] + tokens1 + ["[SEP]"] + tokens2 + ["[SEP]"]
    segment_ids = [0] * (len(tokens1) + 2) \
                  + [1] * (len(tokens2) + 1)
    tokens = tokens[:max_seq_length]
    segment_ids = segment_ids[:max_seq_length]
    return tokens, segment_ids


def combine_with_sep_cls2(max_seq_length, tokens1, tokens2):
    max_seg1_len = max_seq_length - 3 - len(tokens2)
    tokens1 = tokens1[:max_seg1_len]
    tokens = ["[CLS]"] + tokens1 + ["[SEP]"] + tokens2 + ["[SEP]"]
    segment_ids = [0] * (len(tokens1) + 2) \
                  + [1] * (len(tokens2) + 1)
    tokens = tokens[:max_seq_length]
    segment_ids = segment_ids[:max_seq_length]
    return tokens, segment_ids


def concat_triplet_windows(triplet_iterator, window_length=None):
    all_input_ids: List[int] = []
    all_input_mask: List[int] = []
    all_segment_ids: List[int] = []
    for input_ids, input_mask, segment_ids in triplet_iterator:
        all_input_ids.extend(input_ids)
        all_input_mask.extend(input_mask)
        all_segment_ids.extend(segment_ids)
        if window_length is not None:
            assert len(input_ids) == window_length
            assert len(input_mask) == window_length
            assert len(segment_ids) == window_length

    return all_input_ids, all_input_mask, all_segment_ids
