from collections import OrderedDict
from typing import NamedTuple, List

from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator2.segmented_enc.segmented_tfrecord_gen import encode_triplet, encode_seq_prediction

from dataset_specific.mnli.mnli_reader import NLIPairData
from tlm.data_gen.base import combine_with_sep_cls, get_basic_input_feature_as_list, concat_triplet_windows
from trainer_v2.evidence_selector.evidence_candidates import EvidencePair


class HSegmentedPair(NamedTuple):
    p_tokens: List[str]
    h_tokens: List[str]
    st: int
    ed: int
    nli_pair: NLIPairData

    def get_first_h_tokens(self):
        return self.h_tokens[:self.st], self.h_tokens[self.ed:]

    def get_first_h_tokens_w_mask(self):
        return self.h_tokens[:self.st] + ["[MASK]"] + self.h_tokens[self.ed:]

    def get_second_h_tokens(self):
        return self.h_tokens[self.st: self.ed]


class PHSegmentedPair(NamedTuple):
    p_tokens: List[str]
    h_tokens: List[str]
    h_st: int
    h_ed: int
    p_del_indices1: List[int]
    p_del_indices2: List[int]
    nli_pair: NLIPairData

    def get_partial_prem(self, segment_idx: int) -> List[str]:
        assert segment_idx == 0 or segment_idx == 1

        p_tokens_new = list(self.p_tokens)
        del_indices = [self.p_del_indices1, self.p_del_indices2][segment_idx]
        for i in del_indices:
            p_tokens_new[i] = "[MASK]"
        return p_tokens_new

    def get_partial_hypo(self, segment_idx: int) -> List[str]:
        if segment_idx == 0:
            return self.h_tokens[:self.h_st] + ["[MASK]"] + self.h_tokens[self.h_ed:]
        elif segment_idx == 1:
            return self.h_tokens[self.h_st: self.h_ed]
        else:
            raise Exception()


def get_seq_label(del_indices, p_tokens, segment_len) -> List[int]:
    label_v = [0 for _ in p_tokens]
    for j in del_indices:
        label_v[j] = 1

    label_v = [0] + label_v
    pad_n = segment_len - len(label_v)
    output = label_v + [0] * pad_n
    assert len(output) == segment_len
    return output


def concat_ph_to_encode_fn(tokenizer, segment_len, e: PHSegmentedPair):
    triplet_list = []
    for i in [0, 1]:
        tokens, segment_ids = combine_with_sep_cls(
            segment_len, e.get_partial_prem(i), e.get_partial_hypo(i))
        triplet = get_basic_input_feature_as_list(tokenizer, segment_len,
                                                  tokens, segment_ids)
        triplet_list.append(triplet)
    triplet = concat_triplet_windows(triplet_list, segment_len)
    return triplet


# For Training PEP1
def get_ph_segment_pair_encode_fn(segment_len):
    tokenizer = get_tokenizer()

    def encode_fn(e: PHSegmentedPair) -> OrderedDict:
        triplet = concat_ph_to_encode_fn(tokenizer, segment_len, e)
        return encode_triplet(triplet, e.nli_pair.get_label_as_int())

    return encode_fn


# For Training ES1
def get_evidence_pred_encode_fn(segment_len):
    tokenizer = get_tokenizer()

    def encode_fn(e: EvidencePair) -> OrderedDict:
        triplet_list = []
        p_tokens = tokenizer.convert_ids_to_tokens(e.p_tokens)
        label_v_all = []
        for i in [0, 1]:
            h_tokens_i = tokenizer.convert_ids_to_tokens([e.h1, e.h2][i])
            label_v = get_seq_label(e.p_del_indices1, e.p_tokens, segment_len)
            label_v_all.extend(label_v)
            tokens, segment_ids = combine_with_sep_cls(segment_len, p_tokens, h_tokens_i)
            triplet = get_basic_input_feature_as_list(tokenizer, segment_len,
                                                      tokens, segment_ids)
            triplet_list.append(triplet)
        triplet = concat_triplet_windows(triplet_list, segment_len)
        x0, x1, x2 = triplet
        return encode_seq_prediction(x0, x1, x2, label_v_all)
    return encode_fn
