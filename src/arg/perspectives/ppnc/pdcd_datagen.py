from collections import OrderedDict
from typing import List, Dict, Tuple

from arg.perspectives.evaluate import perspective_getter
from arg.perspectives.load import get_claim_perspective_id_dict
from arg.perspectives.ppnc.encode_common import encode_two_inputs
from arg.perspectives.ppnc.ppnc_decl import CPPNCGeneratorInterface, PayloadAsTokens
from data_generator.tokenizer_wo_tf import get_tokenizer, tokenize_from_tokens
from list_lib import left
from tf_util.record_writer_wrap import write_records_w_encode_fn


class Generator(CPPNCGeneratorInterface):
    def __init__(self,
                 cid_to_passages: Dict[int, List[Tuple[List[str], float]]],
                 candidate_perspective: Dict[int, List[int]],
                 ):
        self.gold = get_claim_perspective_id_dict()
        self.candidate_perspective = candidate_perspective
        self.cid_to_passages = cid_to_passages
        self.tokenizer = get_tokenizer()

    def generate_instances(self, claim: Dict, data_id_manager) -> List[PayloadAsTokens]:
        cid = claim['cId']
        claim_tokens = self.tokenizer.tokenize(claim['text'])
        perspectives = self.candidate_perspective[cid]
        passages = self.cid_to_passages[cid]
        output = []
        for pid in perspectives:
            is_correct = any([pid in cluster for cluster in self.gold[cid]])
            perspective = perspective_getter(pid)
            perspective_tokens = self.tokenizer.tokenize(perspective)
            for passage_idx, passage in enumerate(left(passages)):
                passage_subtokens = tokenize_from_tokens(self.tokenizer, passage)
                info = {
                    'cid': cid,
                    'pid': pid,
                    'passage_idx': passage_idx
                }
                p = PayloadAsTokens(passage_subtokens,
                                    perspective_tokens,
                                    claim_tokens,
                                    data_id_manager.assign(info),
                                    is_correct)
                output.append(p)

        return output


def write_records(records: List[PayloadAsTokens],
                  max_seq_length,
                  output_path):
    tokenizer = get_tokenizer()

    def encode(inst: PayloadAsTokens) -> OrderedDict:
        return encode_two_inputs(max_seq_length, tokenizer, inst)

    write_records_w_encode_fn(output_path, encode, records)
