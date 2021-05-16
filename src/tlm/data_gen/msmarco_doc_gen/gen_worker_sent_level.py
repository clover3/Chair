import json
import os
from collections import Counter
from typing import List, Tuple, OrderedDict, Iterable
from typing import List, Iterable, Callable, Dict, Tuple, Set

from arg.qck.decl import QCKQuery, QCKCandidate
from data_generator.job_runner import WorkerInterface
from data_generator.tokenizer_wo_tf import get_tokenizer
from dataset_specific.msmarco.common import load_per_query_docs, MSMarcoDoc
from misc_lib import DataIDManager, tprint, exist_or_mkdir, pick1, print_dict_tab, TimeEstimator
from tf_util.record_writer_wrap import write_records_w_encode_fn
from tlm.data_gen.adhoc_datagen import TitleRepeatInterface
from tlm.data_gen.classification_common import ClassificationInstanceWDataID, \
    write_with_classification_instance_with_id, PairedInstance
from tlm.data_gen.msmarco_doc_gen.misc_common import get_pos_neg_doc_ids_for_qid
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResourceI, ProcessedResourcePredict, \
    ProcessedResourceTitleBodyPredict, ProcessedResourceTitleBodyTrain, ProcessedResourceTitleBodyI
from tlm.data_gen.pairwise_common import combine_features


class PointwiseGenFromText:
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
            for doc_id in self.resource.get_doc_for_query_d()[qid]:
                label = self.resource.get_label(qid, doc_id)
                try:
                    doc = docs_d[doc_id]
                    insts: Iterable[Tuple[List, List]] = self.encoder.encode(q_tokens, doc.title, doc.body)
                    for passage_idx, passage in enumerate(insts):
                        tokens_seg, seg_ids = passage
                        assert type(tokens_seg[0]) == str
                        assert type(seg_ids[0]) == int
                        data_id = data_id_manager.assign({
                            'doc_id': doc_id,
                            'passage_idx': passage_idx,
                            'label': label,
                        })
                        inst = ClassificationInstanceWDataID(tokens_seg, seg_ids, label, data_id)
                        yield inst
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


class PairwiseGenFromText:
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
                insts: List[Tuple[List, List]] = self.encoder.encode(q_tokens, doc.title, doc.body)

                for passage_idx, passage in enumerate(insts):
                    yield passage

            for pos_doc_id in pos_doc_id_list:
                sampled_neg_doc_id = pick1(neg_doc_id_list)
                try:
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
                    success_docs += 1
                except KeyError:
                    missing_cnt += 1
                    missing_doc_qid.append(qid)
                    if missing_cnt > 10:
                        print(missing_doc_qid)
                        print("success: ", success_docs)
                        raise KeyError

    def write(self, insts: List[PairedInstance], out_path: str):
        def encode_fn(inst: PairedInstance) -> OrderedDict:
            return combine_features(inst.tokens1, inst.seg_ids1, inst.tokens2, inst.seg_ids2,
                                    self.tokenizer, self.max_seq_length)
        try:
            length = len(insts)
        except TypeError:
            length = 0

        return write_records_w_encode_fn(out_path, encode_fn, insts, length)


class PredictionGen:
    def __init__(self, resource: ProcessedResourcePredict,
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
        for qid in qids:
            if qid not in self.resource.candidate_doc_d:
                continue

            docs: List[MSMarcoDoc] = load_per_query_docs(qid, None)
            docs_d = {d.doc_id: d for d in docs}

            q_tokens = self.resource.get_q_tokens(qid)
            for doc_id in self.resource.candidate_doc_d[qid]:
                label = self.resource.get_label(qid, doc_id)
                try:
                    doc = docs_d[doc_id]
                    insts: List[Tuple[List, List]] = self.encoder.encode(q_tokens, doc.title, doc.body)
                    for passage_idx, passage in enumerate(insts):
                        tokens_seg, seg_ids = passage
                        assert type(tokens_seg[0]) == str
                        assert type(seg_ids[0]) == int
                        data_id = data_id_manager.assign({
                            'query': QCKQuery(qid, ""),
                            'candidate': QCKCandidate(doc_id, ""),
                            'passage_idx': passage_idx,
                        })
                        inst = ClassificationInstanceWDataID(tokens_seg, seg_ids, label, data_id)
                        yield inst
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

