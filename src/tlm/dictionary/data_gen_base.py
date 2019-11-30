import collections
from functools import partial

from data_generator import tokenizer_wo_tf as tokenization
from tlm.data_gen import bert_data_gen as btd
from tlm.data_gen.base import pad0, get_basic_input_feature, get_masked_lm_features


class Word:
    def __init__(self, subword_tokens):
        self.subword_rep = subword_tokens
        self.word = subword_tokens[0]
        for sw in subword_tokens[1:]:
            assert sw[:2] == "##"
            self.word += sw[2:]


class SegmentInstanceWithDictEntry(object):
    def __init__(self, tokens, segment_ids, dict_word, dict_def, word_loc_list):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = False
        self.dict_word = dict_word
        self.dict_def = dict_def
        self.word_loc_list = word_loc_list

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.tokens]))
        s += "Text : {}\n".format(tokenization.pretty_tokens(self.tokens))
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        if self.dict_word is not None:
            s += "dict_word: %s\n" % self.dict_word.word
        else:
            s += "No word selected\n"
        s += "dict_def : {}\n".format(self.dict_def)
        s += "word loc : {}\n".format(self.word_loc_list)
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


TOKEN_LINE_SEP = "[unused5]"
TOKEN_DEF_SEP = "[unused6]"


def get_dict_input_features(tokenizer, max_def_length, max_d_loc, max_word_len,
                            segment_ids, dict_def, word_loc_list, dict_word):
    d_input_ids = tokenizer.convert_tokens_to_ids(dict_def)
    d_input_ids = d_input_ids[:max_def_length]
    d_input_mask = [1] * len(d_input_ids)

    if word_loc_list:
        target_segment = segment_ids[word_loc_list[0]]
        d_segment_ids = [target_segment] * len(d_input_ids)
    else:
        d_segment_ids = []

    if dict_word is not None:
        selected_word = tokenizer.convert_tokens_to_ids(dict_word.subword_rep)
    else:
        selected_word = []

    d_input_ids = pad0(d_input_ids, max_def_length)
    d_input_mask = pad0(d_input_mask, max_def_length)
    d_location_ids = pad0(word_loc_list[:max_d_loc], max_d_loc)
    d_segment_ids = pad0(d_segment_ids, max_def_length)
    selected_word = pad0(selected_word, max_word_len)

    features = collections.OrderedDict()
    features["d_input_ids"] = btd.create_int_feature(d_input_ids)
    features["d_input_mask"] = btd.create_int_feature(d_input_mask)
    features["d_segment_ids"] = btd.create_int_feature(d_segment_ids)
    features["d_location_ids"] = btd.create_int_feature(d_location_ids)
    features["selected_word"] = btd.create_int_feature(selected_word)
    return features


class DictPredictionEntry(object):
    def __init__(self, tokens, segment_ids, dict_word, dict_def, word_loc_list, masked_lm_positions, masked_lm_labels):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = False
        self.dict_word = dict_word
        self.dict_def = dict_def
        self.word_loc_list = word_loc_list
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.tokens]))
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        s += "dict_word: %s\n" % (" ".join([str(x) for x in self.dict_word.word]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


class OrderedDictBuilder(collections.OrderedDict):
    def extend(self, other_dict):
        for key, value in other_dict.items():
            self.update({key: value})


class DictLMFeaturizer:
    def __init__(self, tokenizer, max_seq_length, max_predictions_per_seq, max_def_length, max_d_loc, max_word_len):
        self.get_basic_input_features = partial(get_basic_input_feature, tokenizer, max_seq_length)
        self.get_masked_lm_features = partial(get_masked_lm_features, tokenizer, max_predictions_per_seq)
        self.get_dict_input_features =\
            partial(get_dict_input_features, tokenizer, max_def_length, max_d_loc, max_word_len)

    def instance_to_features(self, instance):
        basic_features = self.get_basic_input_features(instance.tokens, instance.segment_ids)
        lm_mask_features = self.get_masked_lm_features(instance.masked_lm_positions, instance.masked_lm_labels)
        dict_features = self.get_dict_input_features(instance.segment_ids, instance.dict_def,
                                                     instance.word_loc_list, instance.dict_word)

        next_sentence_label = 1 if instance.is_random_next else 0
        features = OrderedDictBuilder()
        features.extend(basic_features)
        features.extend(lm_mask_features)
        features.extend(dict_features)
        features["next_sentence_labels"] = btd.create_int_feature([next_sentence_label])
        return features


class DictLMFeaturizerUnmasked:
    def __init__(self, tokenizer, max_seq_length, max_def_length, max_d_loc, max_word_len):
        self.get_basic_input_features = partial(get_basic_input_feature, tokenizer, max_seq_length)
        self.get_dict_input_features =\
            partial(get_dict_input_features, tokenizer, max_def_length, max_d_loc, max_word_len)

    def instance_to_features(self, instance):
        basic_features = self.get_basic_input_features(instance.tokens, instance.segment_ids)
        dict_features = self.get_dict_input_features(instance.segment_ids, instance.dict_def,
                                                     instance.word_loc_list, instance.dict_word)

        next_sentence_label = 1 if instance.is_random_next else 0
        features = OrderedDictBuilder()
        features.extend(basic_features)
        features.extend(dict_features)
        features["next_sentence_labels"] = btd.create_int_feature([next_sentence_label])
        return features
