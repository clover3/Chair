import random
from collections import OrderedDict
from typing import NamedTuple, List, Tuple

from data_generator.create_feature import create_int_feature
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import lmap
from tlm.data_gen.base import get_basic_input_feature


class ADAInstance(NamedTuple):
    text: str
    label: int
    domain_id: int
    is_valid_label: int


def combine_source_and_target(source_data: List[Tuple[str, int]],
                 target_data: List[Tuple[str, int]],
                 size_rate: float) -> List[ADAInstance]:

    target_data_size = int(len(source_data) * size_rate)
    if target_data_size > len(target_data):
        random.shuffle(target_data)

    print("source_len={} target_len={}, truncating to {}".format(
        len(source_data),
        len(target_data),
        target_data_size
    ))
    target_data = target_data[:target_data_size]

    source_domain_id = 0
    target_domain_id = 1

    def get_augment_fn(domain_id, drop_label):
        def augment(e: Tuple[str, int]) -> ADAInstance:
            text, label = e
            if drop_label:
                label = 0
                is_valid_label = 0
            else:
                is_valid_label = 1
            return ADAInstance(text, label, domain_id, is_valid_label)
        return augment

    augment_source = get_augment_fn(source_domain_id, False)
    augment_target = get_augment_fn(target_domain_id, True)

    source_insts: List[ADAInstance] = lmap(augment_source, source_data)
    target_insts: List[ADAInstance] = lmap(augment_target, target_data)
    return source_insts + target_insts


def get_encode_fn(max_seq_length):
    tokenizer = get_tokenizer()
    long_count = 0

    def encode(inst: ADAInstance) -> OrderedDict:
        tokens = tokenizer.tokenize(inst.text)
        max_len = max_seq_length - 2
        if len(tokens) > max_len:
            nonlocal long_count
            long_count = long_count + 1
            if long_count > 10:
                print("long text count", long_count)
        tokens = tokens[:max_len]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        seg_ids = [0] * len(tokens)
        feature: OrderedDict = get_basic_input_feature(tokenizer, max_seq_length, tokens, seg_ids)
        feature['label_ids'] = create_int_feature([inst.label])
        feature['domain_ids'] = create_int_feature([inst.domain_id])
        feature['is_valid_label'] = create_int_feature([inst.is_valid_label])
        return feature

    return encode