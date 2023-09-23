from abc import abstractmethod, ABC
from collections import OrderedDict
from typing import List, Iterable, Callable, Dict, Tuple, Set

from transformers import AutoTokenizer

from data_generator.create_feature import create_int_feature
from data_generator2.segmented_enc.seg_encoder_common import PairEncoderInterface
from dataset_specific.mnli.mnli_reader import NLIPairData


class PairEncoderInterfaceHF(ABC):
    # @abstractmethod
    # def encode(self, tokens) -> Tuple[List, List, List]:
    #     # returns input_ids, input_mask, segment_ids
    #     pass

    @abstractmethod
    def encode_from_text(self, text1, text2) -> Tuple[List, List]:
        # returns input_ids, input_mask, segment_ids
        pass


class BasicConcatEncoder(PairEncoderInterface):
    def __init__(self, max_seq_length):
        self.max_seq_length = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def encode_from_text(self, text1, text2) -> Tuple[List, List]:
        encoded_input = self.tokenizer.encode_plus(
            text1,
            text2,
            padding="max_length",
            max_length=self.max_seq_length,
            truncation=True,
        )
        input_ids = encoded_input["input_ids"]
        token_type_ids = encoded_input["token_type_ids"]
        return input_ids, token_type_ids


def get_encode_fn_from_encoder(encoder: PairEncoderInterface):
    def entry_encode(e: NLIPairData) -> Dict:
        features = OrderedDict()
        input_ids, input_mask, segment_ids = encoder.encode_from_text(e.premise, e.hypothesis)
        features["input_ids"] = create_int_feature(input_ids)
        features["input_mask"] = create_int_feature(input_mask)
        features["segment_ids"] = create_int_feature(segment_ids)
        features['label_ids'] = create_int_feature([e.get_label_as_int()])
        return features

    return entry_encode
