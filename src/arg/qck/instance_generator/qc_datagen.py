

from collections import OrderedDict
from typing import List, Iterable, Dict, Callable

from arg.qck.decl import QCKQuery, QCKCandidate, QCInstance, QCInstanceTokenized
from arg.qck.instance_generator.base import InstanceGenerator
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import flatten, lmap
from misc_lib import DataIDManager
from tlm.data_gen.base import get_basic_input_feature
from tlm.data_gen.bert_data_gen import create_int_feature


def enc_to_feature(tokenizer, max_seq_length, inst: QCInstance) -> OrderedDict:
    seg1 = tokenizer.tokenize(inst.query_text)
    seg2 = tokenizer.tokenize(inst.candidate_text)

    input_tokens = ["[CLS]"] + seg1 + ["[SEP]"] + seg2 + ["[SEP]"]
    segment_ids = [0] * (len(seg1) + 2) + [1] * (len(seg2) + 1)

    feature = get_basic_input_feature(tokenizer, max_seq_length, input_tokens, segment_ids)
    feature["data_id"] = create_int_feature([int(inst.data_id)])
    feature["label_ids"] = create_int_feature([int(inst.is_correct)])
    return feature


def enc_to_feature2(tokenizer, max_seq_length, inst: QCInstanceTokenized) -> OrderedDict:
    seg1 = inst.query_text
    seg2 = inst.candidate_text

    input_tokens = ["[CLS]"] + seg1 + ["[SEP]"] + seg2 + ["[SEP]"]
    segment_ids = [0] * (len(seg1) + 2) + [1] * (len(seg2) + 1)

    feature = get_basic_input_feature(tokenizer, max_seq_length, input_tokens, segment_ids)
    feature["data_id"] = create_int_feature([int(inst.data_id)])
    feature["label_ids"] = create_int_feature([int(inst.is_correct)])
    return feature


class QCInstanceGenerator(InstanceGenerator):
    def __init__(self,
                 candidates_dict: Dict[str, List[QCKCandidate]],
                 is_correct_fn: Callable[[QCKQuery, QCKCandidate], bool],
                 ):
        self.max_seq_length = 512
        self.tokenizer = get_tokenizer()
        self.candidates_dict: Dict[str, List[QCKCandidate]] = candidates_dict
        self._is_correct = is_correct_fn

    def generate(self,
                 q_list: Iterable[QCKQuery],
                 data_id_manager: DataIDManager) -> Iterable[QCInstanceTokenized]:
        tokenizer = self.tokenizer
        def convert(query: QCKQuery) -> Iterable[QCInstanceTokenized]:
            candidates = self.candidates_dict[query.query_id]
            for c in candidates:
                info = {
                            'query': query,
                            'candidate': c.light_rep(),
                        }
                yield QCInstanceTokenized(
                    tokenizer.tokenize(query.text),
                    c.get_tokens(tokenizer),
                    data_id_manager.assign(info),
                    self._is_correct(query, c)
                )
        return flatten(lmap(convert, q_list))

    def encode_fn(self, inst: QCInstanceTokenized) -> OrderedDict:
        return enc_to_feature2(self.tokenizer, self.max_seq_length, inst)

