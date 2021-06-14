from collections import OrderedDict
from typing import List, Iterable

from data_generator.tokenizer_wo_tf import get_tokenizer
from misc_lib import pick1, TEL
from tf_util.record_writer_wrap import write_records_w_encode_fn
from tlm.data_gen.classification_common import PairedInstance
from tlm.data_gen.msmarco_doc_gen.bm25_gen.bm25_gen_common import LongSentenceError
from tlm.data_gen.msmarco_doc_gen.misc_common import get_pos_neg_doc_ids_for_qid
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResource10docMulti
from tlm.data_gen.pairwise_common import combine_features


class PairGeneratorFromMultiResource:
    def __init__(self, prcessed_resource: ProcessedResource10docMulti,
                        get_tokens_d_bert,
                        get_tokens_d_bm25,
                        parallel_encoder,
                        max_seq_length):
        self.prcessed_resource = prcessed_resource
        self.get_tokens_d_bert = get_tokens_d_bert
        self.get_tokens_d_bm25 = get_tokens_d_bm25
        self.encoder = parallel_encoder
        self.tokenizer = get_tokenizer()
        self.max_seq_length = max_seq_length

    def generate(self, data_id_manager, qids) -> Iterable[PairedInstance]:
        for qid in TEL(qids):
            if qid not in self.prcessed_resource.get_doc_for_query_d():
                continue
            bert_tokens_d = self.get_tokens_d_bert(qid)
            query_text = self.prcessed_resource.get_query_text(qid)
            bm25_tokens_d = self.get_tokens_d_bm25(qid)
            pos_doc_id_list, neg_doc_id_list \
                = get_pos_neg_doc_ids_for_qid(self.prcessed_resource, qid)

            def iter_passages(doc_id):
                try:
                    bert_title_tokens, bert_doc_tokens = bert_tokens_d[doc_id]
                    bm25_title_tokens, bm25_doc_tokens = bm25_tokens_d[doc_id]
                    segs = self.encoder.encode(
                        query_text,
                        bert_title_tokens, bert_doc_tokens,
                        bm25_title_tokens, bm25_doc_tokens)
                    return segs
                except LongSentenceError:
                    print("LongSentenceError: doc_id={}".format(pos_doc_id))
                    raise

            try:
                for pos_doc_id in pos_doc_id_list:
                    sampled_neg_doc_id = pick1(neg_doc_id_list)
                    for passage_idx1, passage1 in enumerate(iter_passages(pos_doc_id)):
                        for passage_idx2, passage2 in enumerate(iter_passages(sampled_neg_doc_id)):
                            tokens_seg1, seg_ids1 = passage1
                            tokens_seg2, seg_ids2 = passage2

                            data_id = data_id_manager.assign({
                                'doc_id1': pos_doc_id,
                                'passage_idx1': passage_idx1,
                                'doc_id2': sampled_neg_doc_id,
                                'passage_idx2': passage_idx2,
                            })
                            inst = PairedInstance(tokens_seg1, seg_ids1, tokens_seg2, seg_ids2, data_id)
                            yield inst
            except LongSentenceError:
                pass

    def write(self, insts: List[PairedInstance], out_path: str):
        def encode_fn(inst: PairedInstance) -> OrderedDict:
            return combine_features(inst.tokens1, inst.seg_ids1, inst.tokens2, inst.seg_ids2,
                                    self.tokenizer, self.max_seq_length)

        try:
            length = len(insts)
        except TypeError:
            length = 0

        return write_records_w_encode_fn(out_path, encode_fn, insts, length)

