

from collections import OrderedDict
from typing import List, Iterable, Dict, Tuple

from arg.perspectives.ppnc.encode_common import encode_two_inputs
from arg.perspectives.ppnc.ppnc_decl import PayloadAsTokens
from arg.qck.decl import QCKQuery, KDP, QKUnit, QCKInstance, \
    QCKCandidate
from arg.qck.qck_worker import InstanceGenerator
from data_generator.tokenizer_wo_tf import get_tokenizer, tokenize_from_tokens
from list_lib import flatten, lmap
from misc_lib import DataIDManager


class QCKInstanceGenerator(InstanceGenerator):
    def __init__(self,
                 candidates_dict: Dict[str, List[QCKCandidate]],
                 is_correct_fn,
                 ):
        self.max_seq_length = 512
        self.tokenizer = get_tokenizer()
        self.candidates_dict: Dict[str, List[QCKCandidate]] = candidates_dict
        self._is_correct = is_correct_fn

    def generate(self,
                 kc_candidate: Iterable[QKUnit],
                 data_id_manager: DataIDManager,
               ) -> Iterable[QCKInstance]:

        def convert(pair: Tuple[QCKQuery, List[KDP]]) -> Iterable[QCKInstance]:
            query, passages = pair
            candidates = self.candidates_dict[query.query_id]
            for c in candidates:
                for passage in passages:
                    info = {
                                'query': query,
                                'candidate': c,
                                'kdp': passage
                            }
                    yield QCKInstance(
                        query.text,
                        c.text,
                        passage.tokens,
                        data_id_manager.assign(info),
                        self._is_correct(query, c)
                    )

        return flatten(lmap(convert, kc_candidate))

    def _convert_sub_token(self, r: QCKInstance) -> PayloadAsTokens:
        tokenizer = self.tokenizer
        passage_subtokens = tokenize_from_tokens(tokenizer, r.doc_tokens)
        tokens1: List[str] = tokenizer.tokenize(r.query_text)
        tokens2: List[str] = tokenizer.tokenize(r.candidate_text)

        return PayloadAsTokens(passage=passage_subtokens,
                               text1=tokens1,
                               text2=tokens2,
                               data_id=r.data_id,
                               is_correct=r.is_correct,
                               )

    def encode_fn(self, inst: QCKInstance) -> OrderedDict:
        inst_2 = self._convert_sub_token(inst)
        return encode_two_inputs(self.max_seq_length, self.tokenizer, inst_2)

