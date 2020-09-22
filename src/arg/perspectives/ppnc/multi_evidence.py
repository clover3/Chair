import collections
from collections import OrderedDict
from typing import List, Dict, Tuple, NamedTuple

from arg.perspectives.evaluate import perspective_getter
from arg.perspectives.load import get_claim_perspective_id_dict
from arg.perspectives.ppnc.ppnc_decl import CPPNCGeneratorInterface
from data_generator.tokenizer_wo_tf import get_tokenizer, tokenize_from_tokens
from list_lib import left, lmap, lflatten
from tf_util.record_writer_wrap import write_records_w_encode_fn
from tlm.data_gen.base import get_basic_input_feature_as_list, combine_with_sep_cls
from tlm.data_gen.bert_data_gen import create_int_feature


class Payload(NamedTuple):
    passage_list: List[List[str]]
    text1: str
    text2: str
    data_id: int
    is_correct: int


class Generator(CPPNCGeneratorInterface):
    def __init__(self,
                 cid_to_passages: Dict[int, List[Tuple[List[str], float]]],
                 candidate_perspective: Dict[int, List[int]],
                 filer_good
                 ):
        self.gold = get_claim_perspective_id_dict()
        self.candidate_perspective = candidate_perspective
        self.cid_to_passages = cid_to_passages
        self.filter_good = filer_good

    def generate_instances(self, claim: Dict, data_id_manager) -> List[Payload]:
        cid = claim['cId']
        claim = claim['text']
        perspectives = self.candidate_perspective[cid]
        passages = self.cid_to_passages[cid]

        output = []
        for pid in perspectives:
            info = {
                'cid': cid,
                'pid': pid,
            }
            is_correct = any([pid in cluster for cluster in self.gold[cid]])
            perspective = perspective_getter(pid)
            passage_list = left(passages)
            payload = Payload(
                passage_list,
                claim,
                perspective,
                data_id_manager.assign(info),
                is_correct,
            )
            output.append(payload)

        return output


def write_records(records: List[Payload],
                  max_seq_length,
                  d_max_seq_length,
                  output_path):
    tokenizer = get_tokenizer()

    num_windows = int(d_max_seq_length / max_seq_length)

    def combine_and_pad(tokens1, tokens2):
        tokens, segment_ids = combine_with_sep_cls(max_seq_length, tokens1, tokens2)

        pad_len = max_seq_length - len(tokens)
        tokens = tokens + ["[PAD]"] * pad_len
        segment_ids = segment_ids + [0] * pad_len
        return tokens, segment_ids

    def encode(inst: Payload) -> OrderedDict:
        tokens_1_1: List[str] = tokenizer.tokenize(inst.text1)
        tokens_1_2: List[str] = tokenizer.tokenize(inst.text2)

        def tokenize_from_tokens_fn(tokens):
            return tokenize_from_tokens(tokenizer, tokens)

        tokens_2_list: List[List[str]] = lmap(tokenize_from_tokens_fn, inst.passage_list)

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

    write_records_w_encode_fn(output_path, encode, records)
