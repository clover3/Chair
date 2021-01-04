from collections import OrderedDict
from typing import List, Iterable, Tuple

from arg.qck.decl import QCKQuery, KDP, QKRegressionInstance, QKUnit
from arg.qck.instance_generator.base import InstanceGenerator
from data_generator.create_feature import create_int_feature
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import flatten, lmap
from misc_lib import DataIDManager
from tlm.data_gen.base import get_basic_input_feature
from tlm.data_gen.bert_data_gen import create_float_feature


class QKRegressionInstanceGenerator(InstanceGenerator):
    def __init__(self, get_score):
        self.max_seq_length = 512
        self.tokenizer = get_tokenizer()
        self.get_score = get_score

    def generate(self,
                   kc_candidate: Iterable[QKUnit],
                   data_id_manager: DataIDManager,
                   ) -> Iterable[QKRegressionInstance]:

        def convert(pair: Tuple[QCKQuery, List[KDP]]) -> Iterable[QKRegressionInstance]:
            query, passages = pair
            for passage in passages:
                info = {
                            'query': query,
                            'kdp': passage
                        }
                yield QKRegressionInstance(query.text, passage.tokens,
                                 data_id_manager.assign(info),
                                 self.get_score(query, passage)
                                 )

        return flatten(lmap(convert, kc_candidate))

    def tokenize_from_tokens(self, tokens: List[str]) -> List[str]:
        output = []
        for t in tokens:
            ts = self.tokenizer.tokenize(t)
            output.extend(ts)
        return output

    def encode_fn(self, inst: QKRegressionInstance) -> OrderedDict:
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
        features['label_ids'] = create_float_feature([inst.score])
        features['data_id'] = create_int_feature([inst.data_id])
        return features
