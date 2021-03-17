from collections import OrderedDict
from typing import List, Iterable, Dict, Tuple, NamedTuple

from arg.qck.decl import QCKQuery, KDP, QKUnit, PayloadAsTokens, get_light_qckquery, get_light_qckcandidate, \
    get_light_kdp
from arg.qck.encode_common import encode_two_inputs
from arg.qck.instance_generator.base import InstanceGenerator
from arg.qck.instance_generator.qcknc_datagen import QCKCandidateI
from data_generator.tokenizer_wo_tf import get_tokenizer, tokenize_from_tokens
from list_lib import flatten, lmap, dict_value_map
from misc_lib import DataIDManager
from tlm.data_gen.bert_data_gen import create_float_feature
from trec.types import TrecRankedListEntry


class Payload(NamedTuple):
    passage: List[str]
    text1: List[str]
    text2: List[str]
    data_id: int
    is_correct: int
    kdp_score: float


def get_payload_as_token(payload: Payload):
    return PayloadAsTokens(payload.passage, payload.text1, payload.text2, payload.data_id, payload.is_correct)


def get_d_from_ranked_list(rl: List[TrecRankedListEntry]):
    return {e.doc_id: e.score for e in rl}


class QCKInstGenWScore(InstanceGenerator):
    def __init__(self,
                 candidates_dict: Dict[str, List[QCKCandidateI]],
                 is_correct_fn,
                 rel_ranked_list: Dict[str, List[TrecRankedListEntry]],
                 kdp_as_sub_token=False
                 ):
        self.max_seq_length = 512
        self.tokenizer = get_tokenizer()
        self.candidates_dict: Dict[str, List[QCKCandidateI]] = candidates_dict
        self._is_correct = is_correct_fn
        self.kdp_as_sub_token = kdp_as_sub_token
        self.kdp_score_d: Dict[str, Dict[str, float]] = dict_value_map(get_d_from_ranked_list, rel_ranked_list)

    def generate(self,
                 kc_candidate: Iterable[QKUnit],
                 data_id_manager: DataIDManager,
               ) -> Iterable[Payload]:

        def convert(pair: Tuple[QCKQuery, List[KDP]]) -> Iterable[Payload]:
            query, kdp_list = pair
            tokenizer = self.tokenizer
            q_tokens: List[str] = tokenizer.tokenize(query.text)
            candidates = self.candidates_dict[query.query_id]
            num_inst_expectation = len(kdp_list) * len(candidates)
            if num_inst_expectation > 1000 * 1000:
                print(query)
                print(len(kdp_list))
                print(len(candidates))
            p_sub_tokens= []
            for p_idx, kdp in enumerate(kdp_list):
                if self.kdp_as_sub_token:
                    passage_subtokens = kdp.tokens
                else:
                    passage_subtokens = tokenize_from_tokens(tokenizer, kdp.tokens)
                p_sub_tokens.append(passage_subtokens)

            for c in candidates:
                c_tokens: List[str] = c.get_tokens(tokenizer)
                for p_idx, kdp in enumerate(kdp_list):
                    info = {
                                'query': get_light_qckquery(query),
                                'candidate': get_light_qckcandidate(c),
                                'kdp': get_light_kdp(kdp)
                            }
                    passage_subtokens = p_sub_tokens[p_idx]
                    inst = Payload(
                        passage=passage_subtokens,
                        text1=q_tokens,
                        text2=c_tokens,
                        data_id=data_id_manager.assign(info),
                        is_correct=self._is_correct(query, c),
                        kdp_score=self.get_rel_score(query, kdp),
                    )
                    yield inst

        return flatten(lmap(convert, kc_candidate))

    def get_rel_score(self, query: QCKQuery, kdp: KDP) -> float:
        return self.kdp_score_d[query.query_id][kdp.doc_id]

    def encode_fn(self, inst: Payload) -> OrderedDict:
        return encode_two_inputs_w_score(self.max_seq_length, self.tokenizer, inst)


def encode_two_inputs_w_score(max_seq_length, tokenizer, inst: Payload) -> OrderedDict:
    features = encode_two_inputs(max_seq_length, tokenizer, get_payload_as_token(inst))
    features['rel_score'] = create_float_feature([inst.kdp_score])
    return features
