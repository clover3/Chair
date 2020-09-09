from collections import OrderedDict
from typing import List, Dict, Tuple, NamedTuple

from arg.perspectives.evaluate import perspective_getter
from arg.perspectives.load import get_claim_perspective_id_dict
from arg.perspectives.ppnc.encode_common import encode_two_inputs
from arg.perspectives.ppnc.ppnc_decl import CPPNCGeneratorInterface, PayloadAsTokens
from data_generator.tokenizer_wo_tf import get_tokenizer, tokenize_from_tokens
from list_lib import lfilter, left
from tf_util.record_writer_wrap import write_records_w_encode_fn


def score_over_zero(passage_and_score: Tuple[List[str], float]):
    _, score = passage_and_score
    return score > 0


class Payload(NamedTuple):
    passage: List[str]
    text1: str
    text2: str
    data_id: int
    is_correct: int


class Generator(CPPNCGeneratorInterface):
    def __init__(self,
                 cid_to_passages: Dict[int, List[Tuple[List[str], float]]],
                 candidate_perspective: Dict[int, List[int]],
                 filer_good
                 ):
        self.gold = get_claim_perspective_id_dict()
        self.candidate_perspective = candidate_perspective
        self.cid_to_passages = cid_to_passages
        self.filter_good = filer_good

    def generate_instances(self, claim: Dict, data_id_manager) -> List[Payload]:
        cid = claim['cId']
        claim = claim['text']
        perspectives = self.candidate_perspective[cid]
        passages = self.cid_to_passages[cid]

        if self.filter_good:
            filter_condition = score_over_zero
        else:
            def filter_condition(dummy):
                return True
        good_passages: List[List[str]] = left(lfilter(filter_condition, passages))
        output = []
        for pid in perspectives:
            is_correct = any([pid in cluster for cluster in self.gold[cid]])
            for passage_idx, passage in enumerate(good_passages):
                perspective = perspective_getter(pid)
                info = {
                    'cid': cid,
                    'pid': pid,
                    'passage_idx': passage_idx
                }
                p = Payload(passage,
                            claim,
                            perspective,
                            data_id_manager.assign(info),
                            is_correct)
                output.append(p)

        return output


def convert_sub_token(tokenizer, r: Payload) -> PayloadAsTokens:
    passage_subtokens = tokenize_from_tokens(tokenizer, r.passage)
    tokens1: List[str] = tokenizer.tokenize(r.text1)
    tokens2: List[str] = tokenizer.tokenize(r.text2)

    return PayloadAsTokens(passage=passage_subtokens,
                           text1=tokens1,
                           text2=tokens2,
                           data_id=r.data_id,
                           is_correct=r.is_correct
                           )


def write_records(records: List[Payload],
                  max_seq_length,
                  output_path):
    tokenizer = get_tokenizer()

    def encode(inst: Payload) -> OrderedDict:
        inst_2 = convert_sub_token(tokenizer, inst)
        return encode_two_inputs(max_seq_length, tokenizer, inst_2)

    write_records_w_encode_fn(output_path, encode, records)
