from typing import List, Dict, Tuple

from data_generator.tokenizer_wo_tf import get_tokenizer
from misc_lib import pick1, TimeEstimator
from tf_util.record_writer_wrap import write_records_w_encode_fn
from tlm.data_gen.msmarco_doc_gen.misc_common import get_pos_neg_doc_ids_for_qid
from tlm.data_gen.msmarco_doc_gen.mmd_gen_common import MMDGenI
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResourceTitleBodyTokensListI
from tlm.data_gen.query_document_encoder import QueryDocumentEncoderI
from tlm.data_gen.rank_common import QueryDocPairInstance, encode_query_doc_pair_instance


class PairwiseQueryDocGenFromTokensList(MMDGenI):
    def __init__(self, resource: ProcessedResourceTitleBodyTokensListI,
                 encoder: QueryDocumentEncoderI,
                 max_seq_length):
        self.resource = resource
        self.encoder = encoder
        self.tokenizer = get_tokenizer()
        self.max_seq_length = max_seq_length

    def generate(self, data_id_manager, qids):
        missing_cnt = 0
        success_docs = 0
        missing_doc_qid = []
        ticker = TimeEstimator(len(qids))
        for qid in qids:
            if qid not in self.resource.get_doc_for_query_d():
                continue
            ticker.tick()
            tokens_d: Dict[str, Tuple[List[str], List[List[str]]]] = self.resource.get_doc_tokens_d(qid)
            pos_doc_id_list, neg_doc_id_list \
                = get_pos_neg_doc_ids_for_qid(self.resource, qid)
            q_tokens = self.resource.get_q_tokens(qid)

            def iter_passages(doc_id):
                title, body = tokens_d[doc_id]
                # Pair of [Query, Content]
                insts: List[Tuple[List, List]] = self.encoder.encode(q_tokens, title, body)
                return insts

            for pos_doc_id in pos_doc_id_list:
                sampled_neg_doc_id = pick1(neg_doc_id_list)
                try:
                    for passage_idx1, passage1 in enumerate(iter_passages(pos_doc_id)):
                        for passage_idx2, passage2 in enumerate(iter_passages(sampled_neg_doc_id)):
                            q1, d1 = passage1
                            q2, d2 = passage2
                            for t1, t2 in zip(q1, q2):
                                assert t1 == t2
                            assert type(q1[0]) == str
                            assert type(d2[0]) == str
                            data_id = data_id_manager.assign({
                                'qid': qid,
                                'doc_id1': pos_doc_id,
                                'passage_idx1': passage_idx1,
                                'doc_id2': sampled_neg_doc_id,
                                'passage_idx2': passage_idx2,
                            })
                            inst = QueryDocPairInstance(q1, d1, d2, data_id)
                            yield inst
                    success_docs += 1
                except KeyError:
                    missing_cnt += 1
                    missing_doc_qid.append(qid)
                    if missing_cnt > 10:
                        print(missing_doc_qid)
                        print("success: ", success_docs)
                        raise KeyError

    def write(self, insts: List[QueryDocPairInstance], out_path: str):
        def encode_fn(inst: QueryDocPairInstance):
            return encode_query_doc_pair_instance(self.tokenizer, inst)
        return write_records_w_encode_fn(out_path, encode_fn, insts)


