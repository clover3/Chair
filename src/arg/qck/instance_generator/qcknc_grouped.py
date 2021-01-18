import collections
from collections import OrderedDict
from typing import Iterable
from typing import List, Dict, Tuple, NamedTuple

from arg.qck.decl import QKUnit, get_light_qckquery, get_light_qckcandidate, get_light_kdp, QCKCandidateWToken, \
    add_tokens_to_qk_unit, QCKQueryWToken, KDPWToken, QKUnitWToken
from arg.qck.instance_generator.base import InstanceGenerator
from arg.qck.instance_generator.qcknc_datagen import QCKCandidateI
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import lflatten
from list_lib import lmap
from misc_lib import DataIDManager
from tlm.data_gen.base import get_basic_input_feature_as_list, combine_with_sep_cls
from tlm.data_gen.bert_data_gen import create_int_feature


class Payload(NamedTuple):
    kdp_list: List[List[str]]
    text1: List[str]
    text2: List[str]
    data_id: int
    is_correct: int


class QCKGeneratorGrouped(InstanceGenerator):
    def __init__(self,
                 candidates_dict: Dict[str, List[QCKCandidateI]],
                 is_correct_fn,
                 kdp_as_sub_token=False,
                 k_group_size=32,
                 ):
        self.max_seq_length = 512
        self.tokenizer = get_tokenizer()
        self.candidates_dict: Dict[str, List[QCKCandidateI]] = candidates_dict
        self._is_correct = is_correct_fn
        self.kdp_as_sub_token = kdp_as_sub_token
        self.k_group_size = k_group_size

    def generate(self,
                 qk_units: List[QKUnit],
                 data_id_manager: DataIDManager,
                 ) -> List[Payload]:

        def add_tokens_to_qk_unit_fn(qk_unit) -> QKUnitWToken:
            return add_tokens_to_qk_unit(qk_unit, self.tokenizer)

        qk_units_w_tokens: List[QKUnitWToken] = lmap(add_tokens_to_qk_unit_fn, qk_units)

        def convert(target_pair: Tuple[QCKQueryWToken, List[KDPWToken]],
                    ) -> Iterable[Payload]:
            target_query, target_kdp_list = target_pair
            candidates = self.candidates_dict[target_query.query_id]
            candidates_w_tokens = [QCKCandidateWToken.from_qck_candidate(self.tokenizer, c) for c in candidates]
            num_inst_expectation = len(target_kdp_list) * len(candidates)
            if num_inst_expectation > 1000 * 1000:
                print(target_query)
                print(len(target_kdp_list))
                print(len(candidates))

            def get_insts_per_candidate(candidate: QCKCandidateWToken,
                                        query: QCKQueryWToken,
                                        kdp_list: List[KDPWToken]
                                        ) -> Payload:
                kdp_list = kdp_list[:self.k_group_size]

                kdp_token_list = []
                for p_idx, kdp in enumerate(kdp_list):
                    kdp_token_list.append(kdp.sub_tokens)

                info = {
                    'query': get_light_qckquery(query),
                    'candidate': get_light_qckcandidate(candidate),
                    'kdpl': lmap(get_light_kdp, kdp_list)
                }
                inst = Payload(
                    kdp_list=kdp_token_list,
                    text1=query.tokens,
                    text2=candidate.tokens,
                    data_id=data_id_manager.assign(info),
                    is_correct=self._is_correct(query, candidate)
                )
                return inst

            for c_w_token in candidates_w_tokens:
                yield get_insts_per_candidate(c_w_token, target_query, target_kdp_list)

        output: List[Payload] = []
        for idx, pair in enumerate(qk_units_w_tokens):
            output.extend(list(convert(pair)))

        return output

    def encode_fn(self, inst: Payload) -> OrderedDict:
        return encode_multi(self.max_seq_length, self.tokenizer, self.k_group_size, inst)


def encode_multi(max_seq_length, tokenizer, num_windows, inst: Payload) -> OrderedDict:
    d_max_seq_length = max_seq_length * num_windows

    def combine_and_pad(tokens1, tokens2):
        tokens, segment_ids = combine_with_sep_cls(max_seq_length, tokens1, tokens2)

        pad_len = max_seq_length - len(tokens)
        tokens = tokens + ["[PAD]"] * pad_len
        segment_ids = segment_ids + [0] * pad_len
        return tokens, segment_ids

    tokens_1_1: List[str] = inst.text1
    tokens_1_2: List[str] = inst.text2

    tokens_2_list: List[List[str]] = inst.kdp_list

    tokens, segment_ids = combine_with_sep_cls(max_seq_length, tokens_1_1, tokens_1_2)
    input_ids, input_mask, segment_ids = get_basic_input_feature_as_list(tokenizer, max_seq_length,
                                                                         tokens, segment_ids)
    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)

    def iterate_over(tokens1, tokens2_list) -> Tuple[List[str], List[int]]:
        dummy_tokens = ["[PAD]"] * max_seq_length
        dummy_segment_ids = [0] * max_seq_length

        def make_for_each_window(tokens2):
            tokens, segment_ids = combine_and_pad(tokens1, tokens2)
            return tokens, segment_ids

        tokens_and_segment_ids_list: List[Tuple[List[str], List[int]]] = \
            lmap(make_for_each_window, tokens2_list[:num_windows])

        pad_len = num_windows - len(tokens_and_segment_ids_list)
        tokens_and_segment_ids_list += [(dummy_tokens, dummy_segment_ids)] * pad_len
        tokens_list, segment_ids_list = zip(*tokens_and_segment_ids_list)
        return lflatten(tokens_list), lflatten(segment_ids_list)

    def get_second_feature_parts(tokens1, tokens2_list):
        tokens, segment_ids = iterate_over(tokens1, tokens2_list)
        return get_basic_input_feature_as_list(tokenizer, d_max_seq_length, tokens, segment_ids)

    input_ids, input_mask, segment_ids = get_second_feature_parts(tokens_1_2, tokens_2_list)
    features["input_ids2"] = create_int_feature(input_ids)
    features["input_mask2"] = create_int_feature(input_mask)
    features["segment_ids2"] = create_int_feature(segment_ids)

    input_ids, input_mask, segment_ids = get_second_feature_parts(tokens_1_1, tokens_2_list)
    features["input_ids3"] = create_int_feature(input_ids)
    features["input_mask3"] = create_int_feature(input_mask)
    features["segment_ids3"] = create_int_feature(segment_ids)
    features['label_ids'] = create_int_feature([inst.is_correct])
    features['data_id'] = create_int_feature([inst.data_id])
    return features
