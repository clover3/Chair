from collections import OrderedDict
from typing import Iterable, Tuple

from data_generator.create_feature import create_int_feature
from data_generator.tokenizer_wo_tf import get_tokenizer
from tlm.data_gen.adhoc_datagen import FirstSegmentAsDoc
from tlm.data_gen.base import get_basic_input_feature


def encode_classification_feature(max_seq_length, data: Iterable[Tuple[str, str, int]]) -> Iterable[OrderedDict]:
    tokenizer = get_tokenizer()
    encoder = FirstSegmentAsDoc(max_seq_length)
    for query, text, label in data:
        q_tokens = tokenizer.tokenize(query)
        text_tokens = tokenizer.tokenize(text)
        input_tokens, segment_ids = encoder.encode(q_tokens, text_tokens)[0]
        feature = get_basic_input_feature(tokenizer, max_seq_length, input_tokens, segment_ids)
        feature['label_ids'] = create_int_feature([label])
        yield feature