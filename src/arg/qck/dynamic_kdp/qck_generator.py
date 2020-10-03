

from collections import OrderedDict
from typing import List, Iterable, Dict

from arg.qck.decl import QCKQuery, KDP, QCKInstance, \
    QCKCandidate, PayloadAsTokens, QCKCandidateWToken, QCKQueryWToken, KnowledgeDocumentPart
from arg.qck.encode_common import encode_two_inputs
from arg.qck.qck_worker import InstanceGenerator
from data_generator.tokenizer_wo_tf import get_tokenizer, tokenize_from_tokens
from list_lib import flatten, lmap, dict_value_map
from misc_lib import DataIDManager


def light_query(obj: QCKQueryWToken):
    return QCKQuery(obj.query_id, "")


def light_candidate(obj: QCKCandidateWToken):
    return QCKCandidate(obj.id, "")


def light_kdp(obj: KDP) -> KDP:
    return KnowledgeDocumentPart(doc_id=obj.doc_id,
                                   passage_idx=obj.passage_idx,
                                   start_location=obj.start_location,
                                   tokens=[])


class QCKGenDynamicKDP(InstanceGenerator):
    def __init__(self,
                 queries: List[QCKQuery],
                 candidates_dict: Dict[str, List[QCKCandidate]],
                 is_correct_fn,
                 ):
        self.max_seq_length = 512
        self.tokenizer = get_tokenizer()

        def c_list_convert(l: List[QCKCandidate]):
            return lmap(self.get_qck_candidate_w_token, l)

        self.candidates_dict: Dict[str, List[QCKCandidateWToken]] = \
            dict_value_map(c_list_convert, candidates_dict)
        self._is_correct = is_correct_fn
        self.queries: List[QCKQueryWToken] = lmap(self.get_qck_query_w_token, queries)
        print("{} insts will made for each kdp".format(self.num_insts_per_kdp()))

    def get_qck_candidate_w_token(self, c: QCKCandidate) -> QCKCandidateWToken:
        tokens = self.tokenizer.tokenize(c.text)
        return QCKCandidateWToken(c.id, c.text, tokens)

    def get_qck_query_w_token(self, q: QCKQuery) -> QCKQueryWToken:
        tokens = self.tokenizer.tokenize(q.text)
        return QCKQueryWToken(q.query_id, q.text, tokens)

    def num_insts_per_kdp(self):
        cnt = 0
        for query in self.queries:
            cnt += len(self.candidates_dict[query.query_id])
        return cnt

    def generate(self,
                 k_list: List[KDP],
                 data_id_manager: DataIDManager,
               ) -> Iterable[PayloadAsTokens]:

        def convert(k: KDP) -> Iterable[PayloadAsTokens]:
            k_tokens = tokenize_from_tokens(self.tokenizer, k.tokens)
            for query in self.queries:
                for c in self.candidates_dict[query.query_id]:
                    info = {
                                'query': light_query(query),
                                'candidate': light_candidate(c),
                                'kdp': light_kdp(k)
                            }

                    yield PayloadAsTokens(
                        query.tokens,
                        c.tokens,
                        k_tokens,
                        data_id_manager.assign(info),
                        self._is_correct(query, c)
                    )
        return flatten(lmap(convert, k_list))

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

