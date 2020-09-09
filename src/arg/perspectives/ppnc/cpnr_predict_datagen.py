from collections import OrderedDict
from collections import OrderedDict
from typing import List, Iterable, NamedTuple

from arg.perspectives.ppnc.decl import ClaimPassages
from data_generator.create_feature import create_int_feature
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import flatten
from misc_lib import DataIDManager
from tlm.data_gen.base import get_basic_input_feature


class Instance(NamedTuple):
    query_text: str
    passage: List[str]
    data_id: int


def generate_instances(claim_passages_list: Iterable[ClaimPassages],
                       data_id_manager: DataIDManager) -> Iterable[Instance]:

    def convert(pair: ClaimPassages) -> Iterable[Instance]:
        claim, passages = pair
        cid = claim['cId']
        query_text = claim['text']
        for passage_idx, (passage, dummy_score) in enumerate(passages):
            info = {
                        'cid': cid,
                        'passage_idx': passage_idx
                    }
            yield Instance(query_text, passage, data_id_manager.assign(info))

    return flatten(map(convert, claim_passages_list))


def get_encode_fn(max_seq_length):
    tokenizer = get_tokenizer()

    def tokenize_from_tokens(tokens: List[str]) -> List[str]:
        output = []
        for t in tokens:
            ts = tokenizer.tokenize(t)
            output.extend(ts)
        return output

    def encode(inst: Instance) -> OrderedDict:
        tokens1: List[str] = tokenizer.tokenize(inst.query_text)
        max_seg2_len = max_seq_length - 3 - len(tokens1)

        tokens2 = tokenize_from_tokens(inst.passage)[:max_seg2_len]
        tokens = ["[CLS]"] + tokens1 + ["[SEP]"] + tokens2 + ["[SEP]"]

        segment_ids = [0] * (len(tokens1) + 2) \
                      + [1] * (len(tokens2) + 1)
        tokens = tokens[:max_seq_length]
        segment_ids = segment_ids[:max_seq_length]
        features = get_basic_input_feature(tokenizer, max_seq_length, tokens, segment_ids)
        features['label_ids'] = create_int_feature([0])
        features['data_id'] = create_int_feature([inst.data_id])
        return features

    return encode