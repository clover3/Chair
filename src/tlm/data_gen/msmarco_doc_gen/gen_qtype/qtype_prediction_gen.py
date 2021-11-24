import collections
from collections import OrderedDict
from typing import List, Iterable, Dict, Tuple, NamedTuple

from data_generator.create_feature import create_int_feature
from data_generator.tokenizer_wo_tf import get_tokenizer
from misc_lib import TEL
from tf_util.record_writer_wrap import write_records_w_encode_fn
from tlm.data_gen.msmarco_doc_gen.mmd_gen_common import MMDGenI
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResourceTitleBodyTokensListI


class Instance(NamedTuple):
    tokens: List[str]
    qtype_id: int
    data_id: int

class QTypeIDPredictionGen(MMDGenI):
    def __init__(self,
                 resource: ProcessedResourceTitleBodyTokensListI,
                 max_seq_length,
                 qid_to_entity_tokens: Dict[str, List[str]],
                 qid_to_qtype_id: Dict[str, int]
                 ):
        self.resource = resource
        self.tokenizer = get_tokenizer()
        self.qid_to_entity_tokens: Dict[str, List[str]] = qid_to_entity_tokens
        self.max_seq_length = max_seq_length
        self.qid_to_qtype_id = qid_to_qtype_id

    def generate(self, data_id_manager, qids) -> Iterable[Tuple[List, int]]:
        for qid in TEL(qids):
            if qid not in self.resource.get_doc_for_query_d():
                continue

            if qid not in self.qid_to_entity_tokens:
                continue

            q_tokens = self.resource.get_q_tokens(qid)
            entity_tokens = self.qid_to_entity_tokens[qid]
            assert len(entity_tokens) <= len(q_tokens)
            qtype_id: int = self.qid_to_qtype_id[qid]
            data_id = data_id_manager.assign({
                'qtype_id': qtype_id

            })
            inst = Instance(entity_tokens, qtype_id, data_id)
            yield inst

    def write(self, insts: Iterable[Instance], out_path: str):
        def encode_fn(inst: Instance):
            return encode_qtype_id_prediction(self.tokenizer, self.max_seq_length, inst)
        return write_records_w_encode_fn(out_path, encode_fn, insts)


def encode_qtype_id_prediction(tokenizer, max_seq_length, inst: Instance) -> OrderedDict:
    def token_to_feature(tokens):
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        n_pad = max_seq_length - len(input_ids)
        return create_int_feature(input_ids + [0] * n_pad)

    feature = collections.OrderedDict()
    feature['entity'] = token_to_feature(inst.tokens)
    feature['qtype_id'] = create_int_feature([inst.qtype_id])
    feature['data_id'] = create_int_feature([inst.data_id])
    return feature


