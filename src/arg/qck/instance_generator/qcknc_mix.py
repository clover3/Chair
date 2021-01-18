from collections import OrderedDict
from typing import List, Iterable, Dict, Tuple

from arg.qck.decl import QKUnit, PayloadAsTokens, get_light_qckquery, get_light_qckcandidate, get_light_kdp, \
    QCKCandidateWToken, \
    add_tokens_to_qk_unit, QCKQueryWToken, KDPWToken, QKUnitWToken
from arg.qck.encode_common import encode_two_inputs
from arg.qck.instance_generator.qcknc_datagen import QCKCandidateI
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import lmap
from misc_lib import DataIDManager, pick1


# (q1, d1+, t1+) : 1
# (q1, d1-, t1+) : 0
# (q2, d1+, t2+) : 0


class QCKGeneratorMixed:
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
                 qk_units: List[QKUnit],
                 neg_qk_units: List[QKUnit],
                 data_id_manager: DataIDManager,
                 ) -> List[PayloadAsTokens]:

        def add_tokens_to_qk_unit_fn(qk_unit) -> QKUnitWToken:
            return add_tokens_to_qk_unit(qk_unit, self.tokenizer)

        qk_units_w_tokens: List[QKUnitWToken] = lmap(add_tokens_to_qk_unit_fn, qk_units)
        neg_qk_units_w_tokens: List[QKUnitWToken] = lmap(add_tokens_to_qk_unit_fn, neg_qk_units)

        def convert(target_pair: Tuple[QCKQueryWToken, List[KDPWToken]],
                    other_pairs: List[Tuple[QCKQueryWToken, List[KDPWToken]]]
                    ) -> Iterable[PayloadAsTokens]:
            target_query, target_kdp_list = target_pair
            candidates = self.candidates_dict[target_query.query_id]
            candidates_w_tokens = [QCKCandidateWToken.from_qck_candidate(self.tokenizer, c) for c in candidates]
            num_inst_expectation = len(target_kdp_list) * len(candidates)
            if num_inst_expectation > 1000 * 1000:
                print(target_query)
                print(len(target_kdp_list))
                print(len(candidates))

            def get_insts_per_candidate(candidate: QCKCandidateWToken,
                                        query: QCKQueryWToken,
                                        kdp_list: List[KDPWToken]
                                        ):
                inst_per_candidate = []
                for p_idx, kdp in enumerate(kdp_list):
                    info = {
                        'query': get_light_qckquery(query),
                        'candidate': get_light_qckcandidate(candidate),
                        'kdp': get_light_kdp(kdp)
                    }
                    inst = PayloadAsTokens(
                        passage=kdp.sub_tokens,
                        text1=query.tokens,
                        text2=candidate.tokens,
                        data_id=data_id_manager.assign(info),
                        is_correct=self._is_correct(query, candidate)
                    )
                    inst_per_candidate.append(inst)
                return inst_per_candidate

            for c_w_token in candidates_w_tokens:
                yield from get_insts_per_candidate(c_w_token, target_query, target_kdp_list)
                other_query, other_kdp_list = pick1(other_pairs)
                yield from get_insts_per_candidate(c_w_token, other_query, other_kdp_list)

        output: List[PayloadAsTokens] = []
        for idx, pair in enumerate(qk_units_w_tokens):
            output.extend(convert(pair, neg_qk_units_w_tokens))

        return output

    def encode_fn(self, inst: PayloadAsTokens) -> OrderedDict:
        return encode_two_inputs(self.max_seq_length, self.tokenizer, inst)

