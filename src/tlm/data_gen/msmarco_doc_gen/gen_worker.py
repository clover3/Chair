import abc
import json
import os
import random
from abc import ABC
from typing import List, Dict, Tuple, OrderedDict

from arg.qck.decl import QCKQuery, QCKCandidate
from data_generator.job_runner import WorkerInterface
from data_generator.tokenizer_wo_tf import get_tokenizer
from dataset_specific.msmarco.common import SimpleQrel, QueryID
from dataset_specific.msmarco.common import load_query_group, load_candidate_doc_list_1, load_msmarco_simple_qrels, \
    load_token_d_1, load_queries, top100_doc_ids, load_candidate_doc_list_10, load_token_d_10doc, \
    load_candidate_doc_top50, load_token_d_50doc
from misc_lib import DataIDManager, tprint, exist_or_mkdir, pick1
from tf_util.record_writer_wrap import write_records_w_encode_fn
from tlm.data_gen.classification_common import ClassificationInstanceWDataID, \
    write_with_classification_instance_with_id, PairedInstance
from tlm.data_gen.pairwise_common import combine_features


class ProcessedResourceI(ABC):
    def __init__(self, split):
        pass

    @abc.abstractmethod
    def get_doc_tokens_d(self, qid: QueryID) -> Dict[str, List[str]]:
        pass

    @abc.abstractmethod
    def get_label(self, qid: QueryID, doc_id):
        pass

    @abc.abstractmethod
    def get_q_tokens(self, qid: QueryID):
        pass

    @abc.abstractmethod
    def get_doc_for_query_d(self):
        pass

    @abc.abstractmethod
    def query_in_qrel(self, query_id):
        pass


class ProcessedResource(ProcessedResourceI):
    def __init__(self, split):
        super(ProcessedResource, self).__init__(split)
        query_group: List[List[QueryID]] = load_query_group(split)
        candidate_docs_d: Dict[QueryID, List[str]] = load_candidate_doc_list_1(split)
        qrel: SimpleQrel = load_msmarco_simple_qrels(split)

        self.split = split
        self.queires = dict(load_queries(split))
        self.query_group = query_group
        self.tokenizer = get_tokenizer()
        self.candidate_doc_d: Dict[QueryID, List[str]] = candidate_docs_d
        self.qrel = qrel
        self.tokenizer = get_tokenizer()

    def get_doc_tokens_d(self, qid: QueryID) -> Dict[str, List[str]]:
        return load_token_d_1(self.split, qid)

    def get_label(self, qid: QueryID, doc_id):
        return self.qrel.get_label(qid, doc_id)

    def get_q_tokens(self, qid: QueryID):
        query_text = self.queires[qid]
        return self.tokenizer.tokenize(query_text)

    def query_in_qrel(self, query_id):
        return query_id in self.qrel.qrel_d

    def get_doc_for_query_d(self):
        return self.candidate_doc_d


class ProcessedResource10doc(ProcessedResourceI):
    def __init__(self, split):
        super(ProcessedResource10doc, self).__init__(split)
        query_group: List[List[QueryID]] = load_query_group(split)
        candidate_docs_d: Dict[QueryID, List[str]] = load_candidate_doc_list_10(split)
        qrel: SimpleQrel = load_msmarco_simple_qrels(split)

        self.split = split
        self.queires = dict(load_queries(split))
        self.query_group = query_group
        self.tokenizer = get_tokenizer()
        self.candidate_doc_d: Dict[QueryID, List[str]] = candidate_docs_d
        self.qrel = qrel
        self.tokenizer = get_tokenizer()

    def get_doc_tokens_d(self, qid: QueryID) -> Dict[str, List[str]]:
        return load_token_d_10doc(self.split, qid)

    def get_label(self, qid: QueryID, doc_id):
        return self.qrel.get_label(qid, doc_id)

    def get_q_tokens(self, qid: QueryID):
        query_text = self.queires[qid]
        return self.tokenizer.tokenize(query_text)

    def get_doc_for_query_d(self):
        return self.candidate_doc_d

    def query_in_qrel(self, query_id):
        return query_id in self.qrel.qrel_d


class ProcessedResource50doc(ProcessedResourceI):
    def __init__(self, split):
        super(ProcessedResource50doc, self).__init__(split)
        query_group: List[List[QueryID]] = load_query_group(split)
        candidate_docs_d: Dict[QueryID, List[str]] = load_candidate_doc_top50(split)
        qrel: SimpleQrel = load_msmarco_simple_qrels(split)

        self.split = split
        self.queires = dict(load_queries(split))
        self.query_group = query_group
        self.tokenizer = get_tokenizer()
        self.candidate_doc_d: Dict[QueryID, List[str]] = candidate_docs_d
        self.qrel = qrel
        self.tokenizer = get_tokenizer()

    def get_doc_tokens_d(self, qid: QueryID) -> Dict[str, List[str]]:
        return load_token_d_50doc(self.split, qid)

    def get_label(self, qid: QueryID, doc_id):
        return self.qrel.get_label(qid, doc_id)

    def get_q_tokens(self, qid: QueryID):
        query_text = self.queires[qid]
        return self.tokenizer.tokenize(query_text)

    def get_doc_for_query_d(self):
        return self.candidate_doc_d

    def query_in_qrel(self, query_id):
        return query_id in self.qrel.qrel_d


class ProcessedResourcePredict:
    def __init__(self, split):
        query_group: List[List[QueryID]] = load_query_group(split)
        candidate_docs_d: Dict[QueryID, List[str]] = top100_doc_ids(split)
        qrel: SimpleQrel = load_msmarco_simple_qrels(split)

        self.split = split
        self.queires = dict(load_queries(split))
        self.query_group = query_group
        self.tokenizer = get_tokenizer()
        self.candidate_doc_d: Dict[QueryID, List[str]] = candidate_docs_d
        self.qrel = qrel
        self.tokenizer = get_tokenizer()

    def get_doc_tokens_d(self, qid: QueryID) -> Dict[str, List[str]]:
        return load_token_d_1(self.split, qid)

    def get_label(self, qid: QueryID, doc_id):
        return self.qrel.get_label(qid, doc_id)

    def get_q_tokens(self, qid: QueryID):
        query_text = self.queires[qid]
        return self.tokenizer.tokenize(query_text)


class FirstPassageGenerator:
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
        for qid in qids:
            if qid not in self.resource.get_doc_for_query_d():
                assert not self.resource.query_in_qrel(qid)
                continue

            tokens_d = self.resource.get_doc_tokens_d(qid)
            q_tokens = self.resource.get_q_tokens(qid)
            for doc_id in self.resource.get_doc_for_query_d()[qid]:
                label = self.resource.get_label(qid, doc_id)
                try:
                    doc_tokens = tokens_d[doc_id]
                    insts: List[Tuple[List, List]] = self.encoder.encode(q_tokens, doc_tokens)

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


class FirstPassagePairGenerator:
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
        for qid in qids:
            if qid not in self.resource.get_doc_for_query_d():
                assert not self.resource.query_in_qrel(qid)
                continue

            tokens_d = self.resource.get_doc_tokens_d(qid)
            q_tokens = self.resource.get_q_tokens(qid)

            pos_doc_id_list = []
            neg_doc_id_list = []
            for pos_doc_id in self.resource.get_doc_for_query_d()[qid]:

                label = self.resource.get_label(qid, pos_doc_id)
                if label:
                    pos_doc_id_list.append(pos_doc_id)
                else:
                    neg_doc_id_list.append(pos_doc_id)

            def iter_passages(doc_id):
                doc_tokens = tokens_d[doc_id]
                insts: List[Tuple[List, List]] = self.encoder.encode(q_tokens, doc_tokens)

                for passage_idx, passage in enumerate(insts):
                    yield passage
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
                    success_docs += 1
            except KeyError:
                missing_cnt += 1
                missing_doc_qid.append(qid)
                if missing_cnt > 10:
                    print(missing_doc_qid)
                    raise

    def write(self, insts: List[PairedInstance], out_path: str):
        def encode_fn(inst: PairedInstance) -> OrderedDict:
            return combine_features(inst.tokens1, inst.seg_ids1, inst.tokens2, inst.seg_ids2,
                                    self.tokenizer, self.max_seq_length)

        return write_records_w_encode_fn(out_path, encode_fn, insts, len(insts))


class PredictionAllPassageGenerator:
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
                assert qid not in self.resource.qrel.qrel_d
                continue

            tokens_d = self.resource.get_doc_tokens_d(qid)
            q_tokens = self.resource.get_q_tokens(qid)
            for doc_id in self.resource.candidate_doc_d[qid]:
                label = self.resource.get_label(qid, doc_id)
                try:
                    doc_tokens = tokens_d[doc_id]
                    insts: List[Tuple[List, List]] = self.encoder.encode(q_tokens, doc_tokens)

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


class MMDWorker(WorkerInterface):
    def __init__(self, query_group, generator, out_dir):
        self.out_dir = out_dir
        self.query_group = query_group
        self.generator = generator
        self.info_dir = os.path.join(self.out_dir + "_info")
        exist_or_mkdir(self.info_dir)

    def work(self, job_id):
        qids = self.query_group[job_id]
        data_bin = 100000
        data_id_st = job_id * data_bin
        data_id_ed = data_id_st + data_bin
        data_id_manager = DataIDManager(data_id_st, data_id_ed)
        tprint("generating instances")
        insts = list(self.generator.generate(data_id_manager, qids))
        tprint("{} instances".format(len(insts)))
        out_path = os.path.join(self.out_dir, str(job_id))
        self.generator.write(insts, out_path)

        info_path = os.path.join(self.info_dir, "{}.info".format(job_id))
        json.dump(data_id_manager.id_to_info, open(info_path, "w"))


