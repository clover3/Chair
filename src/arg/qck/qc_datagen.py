

from collections import OrderedDict
from typing import List, Iterable, Dict, Callable

from arg.qck.decl import QCKQuery, QCKCandidate, QCInstance, PayloadAsTokens
from arg.qck.encode_common import encode_two_inputs
from arg.qck.qck_worker import InstanceGenerator
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import flatten, lmap
from misc_lib import DataIDManager


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
                 data_id_manager: DataIDManager) -> Iterable[QCInstance]:
        def convert(query: QCKQuery) -> Iterable[QCInstance]:
            candidates = self.candidates_dict[query.query_id]
            for c in candidates:
                info = {
                            'query': query,
                            'candidate': c,
                        }
                yield QCInstance(
                    query.text,
                    c.text,
                    data_id_manager.assign(info),
                    self._is_correct(query, c)
                )
        return flatten(lmap(convert, q_list))

    def _convert_sub_token(self, r: QCInstance) -> PayloadAsTokens:
        tokenizer = self.tokenizer
        tokens1: List[str] = tokenizer.tokenize(r.query_text)
        tokens2: List[str] = tokenizer.tokenize(r.candidate_text)

        return PayloadAsTokens(text1=tokens1,
                               text2=tokens2,
                               data_id=r.data_id,
                               is_correct=r.is_correct,
                               )

    def encode_fn(self, inst: QCInstance) -> OrderedDict:
        inst_2 = self._convert_sub_token(inst)
        return encode_two_inputs(self.max_seq_length, self.tokenizer, inst_2)

