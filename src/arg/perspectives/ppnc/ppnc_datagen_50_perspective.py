from collections import OrderedDict
from typing import List, Dict, Tuple, NamedTuple

from arg.perspectives.claim_lm.passage_common import score_over_zero
from arg.perspectives.evaluate import perspective_getter
from arg.perspectives.load import get_claim_perspective_id_dict
from arg.perspectives.ppnc.ppnc_decl import CPPNCGeneratorInterface
from data_generator.create_feature import create_int_feature
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import lfilter, left, lmap, foreach
from tf_util.record_writer_wrap import RecordWriterWrap
from tlm.data_gen.base import get_basic_input_feature


class Payload(NamedTuple):
    passage: List[str]
    candidate_text: str
    data_id: int
    is_correct: int


class Generator(CPPNCGeneratorInterface):
    def __init__(self,
                 cid_to_passages: Dict[int, List[Tuple[List[str], float]]],
                 candidate_perspective: Dict[int, List[int]],
                 ):
        self.gold = get_claim_perspective_id_dict()
        self.candidate_perspective = candidate_perspective
        self.cid_to_passages = cid_to_passages

    def generate_instances(self, claim: Dict, data_id_manager) -> List[Payload]:
        cid = claim['cId']
        perspectives = self.candidate_perspective[cid]
        passages = self.cid_to_passages[cid]
        good_passages: List[List[str]] = left(lfilter(score_over_zero, passages))
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
                p = Payload(passage, perspective, data_id_manager.assign(info), is_correct)
                output.append(p)

        return output


def write_records(records: List[Payload],
                  max_seq_length,
                  output_path):
    tokenizer = get_tokenizer()

    def tokenize_from_tokens(tokens: List[str]) -> List[str]:
        output = []
        for t in tokens:
            ts = tokenizer.tokenize(t)
            output.extend(ts)
        return output

    def encode(inst: Payload) -> OrderedDict:
        tokens1: List[str] = tokenizer.tokenize(inst.candidate_text)
        max_seg2_len = max_seq_length - 3 - len(tokens1)
        tokens2 = tokenize_from_tokens(inst.passage)[:max_seg2_len]
        tokens = ["[CLS]"] + tokens1 + ["[SEP]"] + tokens2 + ["[SEP]"]
        segment_ids = [0] * (len(tokens1) + 2) \
                      + [1] * (len(tokens2) + 1)
        tokens = tokens[:max_seq_length]
        segment_ids = segment_ids[:max_seq_length]
        features = get_basic_input_feature(tokenizer, max_seq_length, tokens, segment_ids)
        features['label_ids'] = create_int_feature([inst.is_correct])
        features['data_id'] = create_int_feature([inst.data_id])
        return features

    writer = RecordWriterWrap(output_path)
    features: List[OrderedDict] = lmap(encode, records)
    foreach(writer.write_feature, features)
    writer.close()


