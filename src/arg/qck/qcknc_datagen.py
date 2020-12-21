

from collections import OrderedDict
from typing import List, Iterable, Dict, Tuple, Union

from arg.qck.decl import QCKQuery, KDP, QKUnit, QCKInstance, \
    QCKCandidate, PayloadAsTokens, get_light_qckquery, get_light_qckcandidate, get_light_kdp, QCKCandidateWToken, \
    PayloadAsIds
from arg.qck.encode_common import encode_two_inputs, encode_two_input_ids
from arg.qck.qck_worker import InstanceGenerator
from data_generator.tokenizer_wo_tf import get_tokenizer, tokenize_from_tokens
from list_lib import flatten, lmap
from misc_lib import DataIDManager

QCKCandidateI = Union[QCKCandidate, QCKCandidateWToken]


class QCKInstanceGenerator(InstanceGenerator):
    def __init__(self,
                 candidates_dict: Dict[str, List[QCKCandidateI]],
                 is_correct_fn,
                 kdp_as_sub_token=False
                 ):
        self.max_seq_length = 512
        self.tokenizer = get_tokenizer()
        self.candidates_dict: Dict[str, List[QCKCandidateI]] = candidates_dict
        self._is_correct = is_correct_fn
        self.kdp_as_sub_token = kdp_as_sub_token

    def generate(self,
                 kc_candidate: Iterable[QKUnit],
                 data_id_manager: DataIDManager,
               ) -> Iterable[PayloadAsTokens]:

        def convert(pair: Tuple[QCKQuery, List[KDP]]) -> Iterable[PayloadAsTokens]:
            query, passages = pair
            tokenizer = self.tokenizer
            q_tokens: List[str] = tokenizer.tokenize(query.text)
            candidates = self.candidates_dict[query.query_id]
            num_inst_expectation = len(passages) * len(candidates)
            if num_inst_expectation > 1000 * 1000:
                print(query)
                print(len(passages))
                print(len(candidates))
            p_sub_tokens= []
            for p_idx, passage in enumerate(passages):
                if self.kdp_as_sub_token:
                    passage_subtokens = passage.tokens
                else:
                    passage_subtokens = tokenize_from_tokens(tokenizer, passage.tokens)
                p_sub_tokens.append(passage_subtokens)

            for c in candidates:
                c_tokens: List[str] = c.get_tokens(tokenizer)
                for p_idx, passage in enumerate(passages):
                    info = {
                                'query': get_light_qckquery(query),
                                'candidate': get_light_qckcandidate(c),
                                'kdp': get_light_kdp(passage)
                            }
                    passage_subtokens = p_sub_tokens[p_idx]
                    inst = PayloadAsTokens(
                        passage=passage_subtokens,
                        text1=q_tokens,
                        text2=c_tokens,
                        data_id=data_id_manager.assign(info),
                        is_correct=self._is_correct(query, c)
                    )
                    yield inst

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

    def encode_fn(self, inst: PayloadAsTokens) -> OrderedDict:
        return encode_two_inputs(self.max_seq_length, self.tokenizer, inst)


class QCKInstanceGeneratorIDversion(InstanceGenerator):
    def __init__(self,
                 candidates_dict: Dict[str, List[QCKCandidateI]],
                 is_correct_fn,
                 kdp_as_sub_token=False
                 ):
        self.max_seq_length = 512
        self.tokenizer = get_tokenizer()
        self.candidates_dict: Dict[str, List[QCKCandidateI]] = candidates_dict
        self._is_correct = is_correct_fn
        self.kdp_as_sub_token = kdp_as_sub_token

    def generate(self,
                 kc_candidate: Iterable[QKUnit],
                 data_id_manager: DataIDManager,
               ) -> Iterable[PayloadAsTokens]:

        def convert(pair: Tuple[QCKQuery, List[KDP]]) -> Iterable[PayloadAsTokens]:
            query, passages = pair
            tokenizer = self.tokenizer
            q_tokens: List[int] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(query.text))
            candidates = self.candidates_dict[query.query_id]
            num_inst_expectation = len(passages) * len(candidates)
            if num_inst_expectation > 1000 * 1000:
                print(query)
                print(len(passages))
                print(len(candidates))

            passage_input_ids_list= []
            for p_idx, passage in enumerate(passages):
                if self.kdp_as_sub_token:
                    passage_subtokens = passage.tokens
                else:
                    passage_subtokens = tokenize_from_tokens(tokenizer, passage.tokens)
                passage_input_ids_list.append(tokenizer.convert_tokens_to_ids(passage_subtokens))

            for c in candidates:
                c_tokens: List[int] = tokenizer.convert_tokens_to_ids(c.get_tokens(tokenizer))
                for p_idx, passage in enumerate(passages):
                    info = {
                                'query': get_light_qckquery(query),
                                'candidate': get_light_qckcandidate(c),
                                'kdp': get_light_kdp(passage)
                            }
                    passage_subtokens = passage_input_ids_list[p_idx]
                    inst = PayloadAsIds(
                        passage=passage_subtokens,
                        text1=q_tokens,
                        text2=c_tokens,
                        data_id=data_id_manager.assign(info),
                        is_correct=self._is_correct(query, c)
                    )
                    yield inst

        return flatten(lmap(convert, kc_candidate))

    def encode_fn(self, inst: PayloadAsIds) -> OrderedDict:
        return encode_two_input_ids(self.max_seq_length, self.tokenizer, inst)

