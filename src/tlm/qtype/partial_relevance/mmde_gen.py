from typing import List, Tuple

from arg.qck.decl import QCKQuery, QCKCandidate
from data_generator.tokenizer_wo_tf import get_tokenizer
from dataset_specific.msmarco.common import load_per_query_docs, MSMarcoDoc
from misc_lib import DataIDManager, TEL
from tlm.data_gen.classification_common import ClassificationInstanceWDataID, \
    write_with_classification_instance_with_id
from tlm.data_gen.msmarco_doc_gen.mmd_gen_common import MMDGenI
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResourcePredict


class MMDEPredictionGen(MMDGenI):
    def __init__(self, resource: ProcessedResourcePredict,
                 basic_encoder,
                 max_seq_length):
        self.resource = resource
        self.encoder = basic_encoder
        self.tokenizer = get_tokenizer()
        self.max_seq_length = max_seq_length

    def generate(self, data_id_manager: DataIDManager, qids):
        missing_cnt = 0
        success_docs = 0
        missing_doc_qid = []
        for qid in TEL(qids):
            if qid not in self.resource.candidate_doc_d:
                continue

            docs: List[MSMarcoDoc] = load_per_query_docs(qid, None)
            docs_d = {d.doc_id: d for d in docs}
            q_tokens = self.resource.get_q_tokens(qid)
            doc_ids = self.resource.candidate_doc_d[qid]
            rel_doc_ids = [doc_id for doc_id in doc_ids if self.resource.get_label(qid, doc_id)]
            nonrel_doc_ids = [doc_id for doc_id in doc_ids if not self.resource.get_label(qid, doc_id)]

            n_non_rel = 20 - len(rel_doc_ids)
            nonrel_doc_ids = nonrel_doc_ids[:n_non_rel]
            for doc_id in rel_doc_ids + nonrel_doc_ids:
                label = self.resource.get_label(qid, doc_id)
                try:
                    doc = docs_d[doc_id]
                    insts: List[Tuple[List, List]] = self.encoder.encode(q_tokens, doc.title, doc.body)
                    for passage_idx, passage in enumerate(insts):
                        tokens_seg, seg_ids = passage
                        assert type(tokens_seg[0]) == str
                        assert type(seg_ids[0]) == int
                        data_id = data_id_manager.assign({
                            'query': QCKQuery(qid, q_tokens),
                            'candidate': QCKCandidate(doc_id, ""),
                            'passage_idx': passage_idx,
                            'title': doc.title,
                            'body': doc.body
                        })
                        inst = ClassificationInstanceWDataID(tokens_seg, seg_ids, label, data_id)
                        yield inst
                        break
                    success_docs += 1
                except KeyError:
                    missing_cnt += 1
                    missing_doc_qid.append(qid)
                    if missing_cnt > 10:
                        print(missing_doc_qid)
                        print("success: ", success_docs)
                        raise KeyError

    def write(self, insts: List[ClassificationInstanceWDataID], out_path: str):
        return write_with_classification_instance_with_id(self.tokenizer, self.max_seq_length, insts, out_path)


