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




class MatchNode:
    def __init__(self, token: int):
        self.token = token
        self.child_dict = dict()
        self.child_keys = set()
        self.is_end = False
        self.new_ids = []



class MatchTree:
    def __init__(self):
        self.root = MatchNode(0)
        self.voca_dict = dict()
        self.term_idx = 100
        self.all_tokens = set()
        self.seq_set = []

    def add_seq(self, seq: List[int]):
        assert len(seq) > 0
        cur_node = self.root
        self.seq_set.append(seq)
        new_ids = []
        for t in seq:
            self.all_tokens.add(t)
            self.voca_dict[t] = self.term_idx
            new_ids.append(self.term_idx)
            self.term_idx += 1

        for t in seq:
            if t not in cur_node.child_dict:
                cur_node.child_dict[t] = MatchNode(t)
                cur_node.child_keys.add(t)
            cur_node = cur_node.child_dict[t]

        cur_node.is_end = True
        cur_node.new_ids = new_ids


def convert_alt_emb2(source_path, output_path, match_tree: MatchTree, include_not_match=False):
    min_overlap = min([len(set(tokens)) for tokens in match_tree.seq_set])

    def get_alt_emb(input_ids):
        alt_emb_mask = []
        alt_input_ids = []
        s = set(input_ids)
        if len(s.intersection(match_tree.all_tokens)) < min_overlap:
            return False, alt_emb_mask, alt_input_ids

        any_success = False
        prev_match = []
        prev_success = False
        cur_node = match_tree.root
        for i in range(len(input_ids)):
            if input_ids[i] in cur_node.child_keys:
                prev_match.append(input_ids[i])
                cur_node = cur_node.child_dict[input_ids[i]]
                if cur_node.is_end and len(cur_node.child_keys) == 0:
                    alt_input_ids.extend(cur_node.new_ids)
                    alt_emb_mask.extend([1] * len(cur_node.new_ids))
                    any_success = True
                    prev_match = []
                elif cur_node.is_end and len(cur_node.child_keys) > 0:
                    prev_match = list(cur_node.new_ids)
                    prev_success = True

            else:
                if prev_match:
                    alt_input_ids.extend(prev_match)
                    if prev_success:
                        alt_emb_mask.extend([1] * len(prev_match))
                        any_success = True
                    else:
                        alt_emb_mask.extend([0] * len(prev_match))
                    prev_success = False
                    prev_match = []


                alt_input_ids.append(input_ids[i])
                alt_emb_mask.append(0)

                assert len(alt_input_ids) == i+1
                cur_node = match_tree.root
        assert len(alt_emb_mask) == len(alt_input_ids)

        return any_success, alt_emb_mask, alt_input_ids

    def feature_transformer(feature):
        new_features = collections.OrderedDict()
        success = False
        for key in feature:
            v = take(feature[key])
            if key == "input_ids":
                input_ids = v
                success, alt_emb_mask, alt_input_ids = get_alt_emb(input_ids)
                if not success and include_not_match:
                    assert len(input_ids) > 0
                    alt_emb_mask = [0] * len(input_ids)
                    alt_input_ids = [0] * len(input_ids)

                new_features["alt_emb_mask"] = create_int_feature(alt_emb_mask)
                new_features["alt_input_ids"] = create_int_feature(alt_input_ids)
            new_features[key] = create_int_feature(v)

        if success or include_not_match:
            return new_features
        else:
            return None

    return tfrecord_convertor_with_none(source_path, output_path, feature_transformer)


def verify_alt_emb(source_path, seq_set: List[List[int]]):
    all_tokens: Set[int] = set(flatten(seq_set))

    def check_feature(feature):
        feature_d = {}
        for key in feature:
            v = take(feature[key])
            feature_d[key] = v

        input_ids = feature_d["input_ids"]
        alt_emb_mask = feature_d["alt_emb_mask"]

        for i in range(len(input_ids)):
            if alt_emb_mask[i] and input_ids[i] not in all_tokens:
                print(i, input_ids[i])

    feature_itr = load_record_v2(source_path)
    for feature in feature_itr:
        check_feature(feature)


