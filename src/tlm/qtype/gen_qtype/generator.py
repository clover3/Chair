from collections import OrderedDict
from typing import List, Tuple

from data_generator.tokenizer_wo_tf import get_tokenizer
from dataset_specific.msmarco.common import load_per_query_docs, MSMarcoDoc
from misc_lib import pick1, TimeEstimator
from tf_util.record_writer_wrap import write_records_w_encode_fn
from tlm.data_gen.classification_common import PairedInstance
from tlm.data_gen.msmarco_doc_gen.misc_common import get_pos_neg_doc_ids_for_qid
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResourceI
from tlm.data_gen.pairwise_common import combine_features
from tlm.qtype.gen_qtype.encoders import DropException


class PairwiseGenWDropTokenFromText:
    def __init__(self, resource: ProcessedResourceI,
                 basic_encoder,
                 max_seq_length):
        self.resource = resource
        self.encoder = basic_encoder
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
            docs: List[MSMarcoDoc] = load_per_query_docs(qid, None)
            docs_d = {d.doc_id: d for d in docs}

            q_tokens = self.resource.get_q_tokens(qid)
            pos_doc_id_list, neg_doc_id_list \
                = get_pos_neg_doc_ids_for_qid(self.resource, qid)

            def iter_passages(doc_id):
                doc = docs_d[doc_id]
                insts: List[Tuple[List, List, List, List]] = self.encoder.encode(q_tokens, doc.title, doc.body)

                for passage_idx, passage in enumerate(insts):
                    yield passage

            for pos_doc_id in pos_doc_id_list:
                sampled_neg_doc_id = pick1(neg_doc_id_list)
                try:
                    for passage_idx1, passage1 in enumerate(iter_passages(pos_doc_id)):
                        for passage_idx2, passage2 in enumerate(iter_passages(sampled_neg_doc_id)):
                            t1, seg_ids1, t1_drop, seg_ids1_drop = passage1
                            t2, seg_ids2, t2_drop, seg_ids2_drop = passage2

                            data_id = data_id_manager.assign({
                                'doc_id1': pos_doc_id,
                                'passage_idx1': passage_idx1,
                                'doc_id2': sampled_neg_doc_id,
                                'passage_idx2': passage_idx2,
                            })
                            inst = PairedInstance(t1, seg_ids1, t2, seg_ids2, data_id)
                            inst_drop = PairedInstance(t1_drop, seg_ids1_drop, t2_drop, seg_ids2_drop, data_id)
                            yield inst, inst_drop
                    success_docs += 1
                except DropException:
                    print("Skip query {}".format(" ".join(q_tokens)))
                    pass
                except KeyError:
                    missing_cnt += 1
                    missing_doc_qid.append(qid)
                    if missing_cnt > 10:
                        print(missing_doc_qid)
                        print("success: ", success_docs)
                        raise KeyError

    def write(self, insts: List[Tuple[PairedInstance, PairedInstance]], out_path: str):
        def encode_fn(inst_pair: Tuple[PairedInstance, PairedInstance]) -> OrderedDict:
            inst, inst_drop = inst_pair
            od: OrderedDict = combine_features(inst.tokens1, inst.seg_ids1, inst.tokens2, inst.seg_ids2,
                                               self.tokenizer, self.max_seq_length)
            od_drop: OrderedDict = combine_features(inst_drop.tokens1,
                                                    inst_drop.seg_ids1,
                                                    inst_drop.tokens2,
                                                    inst_drop.seg_ids2,
                                                    self.tokenizer, self.max_seq_length)

            for key, value in od_drop.items():
                od["drop_" + key] = value
            return od
        try:
            length = len(insts)
        except TypeError:
            length = 0

        return write_records_w_encode_fn(out_path, encode_fn, insts, length)
