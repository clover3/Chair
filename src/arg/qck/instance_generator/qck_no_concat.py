from collections import OrderedDict
from typing import List, Dict, Union

from arg.qck.decl import QCKCandidate, PayloadAsTokens, QCKCandidateWToken
from arg.qck.encode_common import encode_three_inputs
from arg.qck.instance_generator.qcknc_datagen import QCKInstanceGenerator

QCKCandidateI = Union[QCKCandidate, QCKCandidateWToken]


class QCKNoConcatInstGen(QCKInstanceGenerator):
    def __init__(self,
                 candidates_dict: Dict[str, List[QCKCandidateI]],
                 is_correct_fn,
                 kdp_as_sub_token=False
                 ):
        super(QCKNoConcatInstGen, self).__init__(candidates_dict, is_correct_fn, kdp_as_sub_token)
        self.max_seq_length_list = [100, 100, 512]

    def encode_fn(self, inst: PayloadAsTokens) -> OrderedDict:
        return encode_three_inputs(self.max_seq_length_list, self.tokenizer, inst)
