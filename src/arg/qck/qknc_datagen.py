from collections import OrderedDict
from typing import List, Iterable, Tuple

from arg.qck.decl import QCKQuery, KDP, QKInstance, QKUnit
from arg.qck.qck_worker import InstanceGenerator
from data_generator.create_feature import create_int_feature
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import flatten, lmap
from misc_lib import DataIDManager
from tlm.data_gen.base import get_basic_input_feature


class QKInstanceGenerator(InstanceGenerator):
    def __init__(self, is_correct_fn):
        self.max_seq_length = 512
        self.tokenizer = get_tokenizer()
        self._is_correct = is_correct_fn

    def generate(self,
                   kc_candidate: Iterable[QKUnit],
                   data_id_manager: DataIDManager,
                   ) -> Iterable[QKInstance]:

        def convert(pair: Tuple[QCKQuery, List[KDP]]) -> Iterable[QKInstance]:
            query, passages = pair
            for passage in passages:
                info = {
                            'query': query,
                            'kdp': passage
                        }
                yield QKInstance(query.text, passage.tokens,
                                 data_id_manager.assign(info),
                                 self._is_correct(query, passage)
                                 )

        return flatten(lmap(convert, kc_candidate))

    def tokenize_from_tokens(self, tokens: List[str]) -> List[str]:
        output = []
        for t in tokens:
            ts = self.tokenizer.tokenize(t)
            output.extend(ts)
        return output

    def encode_fn(self, inst: QKInstance) -> OrderedDict:
        max_seq_length = self.max_seq_length
        tokens1: List[str] = self.tokenizer.tokenize(inst.query_text)
        max_seg2_len = self.max_seq_length - 3 - len(tokens1)

        tokens2 = self.tokenize_from_tokens(inst.doc_tokens)[:max_seg2_len]
        tokens = ["[CLS]"] + tokens1 + ["[SEP]"] + tokens2 + ["[SEP]"]

        segment_ids = [0] * (len(tokens1) + 2) \
                      + [1] * (len(tokens2) + 1)
        tokens = tokens[:max_seq_length]
        segment_ids = segment_ids[:max_seq_length]
        features = get_basic_input_feature(self.tokenizer, max_seq_length, tokens, segment_ids)
        features['label_ids'] = create_int_feature([inst.is_correct])
        features['data_id'] = create_int_feature([inst.data_id])
        return features
