import json
import os
from collections import OrderedDict
from typing import List, Dict, Tuple

from data_generator.job_runner import WorkerInterface
from data_generator.tokenizer_wo_tf import get_tokenizer
from dataset_specific.msmarco.common import QueryID
from dataset_specific.msmarco.passage_to_doc.save_passages_grouped import load_passage_d_for_job
from misc_lib import exist_or_mkdir, DataIDManager, tprint, pick1, Averager
from tf_util.record_writer_wrap import write_records_w_encode_fn
from tlm.data_gen.classification_common import PairedInstance
from tlm.data_gen.pairwise_common import combine_features
from trec.types import QRelsDict


class MMDWorkerForPassageBased(WorkerInterface):
    def __init__(self, query_group, generator, out_dir):
        self.out_dir = out_dir
        self.query_group = query_group
        self.generator = generator
        self.info_dir = os.path.join(self.out_dir + "_info")
        exist_or_mkdir(self.info_dir)

    def work(self, job_id):
        data_bin = 100000
        data_id_st = job_id * data_bin
        data_id_ed = data_id_st + data_bin
        data_id_manager = DataIDManager(data_id_st, data_id_ed)
        tprint("generating instances")
        insts = self.generator.generate(data_id_manager, job_id)
        # tprint("{} instances".format(len(insts)))
        out_path = os.path.join(self.out_dir, str(job_id))
        self.generator.write(insts, out_path)

        info_path = os.path.join(self.info_dir, "{}.info".format(job_id))
        json.dump(data_id_manager.id_to_info, open(info_path, "w"))


class PairGenerator:
    def __init__(self, basic_encoder,
                 query_group,
                 qrel: QRelsDict,
                 queries_d: Dict[str, str],
                 max_seq_length):
        self.encoder = basic_encoder
        self.tokenizer = get_tokenizer()
        self.max_seq_length = max_seq_length
        self.query_group = query_group
        self.queries_d = queries_d
        self.qrel = qrel

    def generate(self, data_id_manager, job_id):
        missing_cnt = 0
        success_docs = 0
        missing_doc_qid = []
        passage_d: Dict[QueryID, List[Tuple[str, str]]] = load_passage_d_for_job(job_id)
        qids = self.query_group[job_id]

        for qid in qids:
            try:
                passages: List[Tuple[str, str]] = passage_d[qid]
                qrel_per_qid = self.qrel[qid]
                q_tokens = self.tokenizer.tokenize(self.queries_d[qid])
                def relevant(pid):
                    return pid in qrel_per_qid and qrel_per_qid[pid]

                def encode(text):
                    doc_tokens = self.tokenizer.tokenize(text)
                    insts: List[Tuple[List, List]] = self.encoder.encode(q_tokens, doc_tokens)
                    return insts[0]

                pos_docs = list([(pid, content) for pid, content in passages if relevant(pid)])
                neg_docs = list([(pid, content) for pid, content in passages if not relevant(pid)])
                if not neg_docs:
                    print(qid)
                    print("len(pos_doc)", len(pos_docs))
                    print("len(neg_doc)", len(neg_docs))
                    raise IndexError
                for pos_pid, pos_content in pos_docs:
                    neg_pid, neg_content = pick1(neg_docs)
                    tokens_seg1, seg_ids1 = encode(pos_content)
                    tokens_seg2, seg_ids2 = encode(neg_content)

                    data_id = data_id_manager.assign({
                        'doc_id1': pos_pid,
                        'passage_idx1': 0,
                        'doc_id2': neg_pid,
                        'passage_idx2': 0,
                    })
                    inst = PairedInstance(tokens_seg1, seg_ids1, tokens_seg2, seg_ids2, data_id)
                    yield inst
                    success_docs += 1
            except KeyError:
                missing_cnt += 1
                missing_doc_qid.append(qid)

    def write(self, insts: List[PairedInstance], out_path: str):
        def encode_fn(inst: PairedInstance) -> OrderedDict:
            return combine_features(inst.tokens1, inst.seg_ids1, inst.tokens2, inst.seg_ids2,
                                    self.tokenizer, self.max_seq_length)

        try:
            length = len(insts)
        except TypeError:
            length = 0

        return write_records_w_encode_fn(out_path, encode_fn, insts, length)



class LengthStat:
    def __init__(self, basic_encoder,
                 query_group,
                 qrel: QRelsDict,
                 queries_d: Dict[str, str],
                 max_seq_length):
        self.encoder = basic_encoder
        self.tokenizer = get_tokenizer()
        self.max_seq_length = max_seq_length
        self.query_group = query_group
        self.queries_d = queries_d
        self.qrel = qrel

    def generate(self, data_id_manager, job_id):
        missing_cnt = 0
        success_docs = 0
        missing_doc_qid = []
        passage_d: Dict[QueryID, List[Tuple[str, str]]] = load_passage_d_for_job(job_id)
        qids = self.query_group[job_id]

        averager = Averager()
        for qid in qids:
            try:
                passages: List[Tuple[str, str]] = passage_d[qid]
                qrel_per_qid = self.qrel[qid]
                q_tokens = self.tokenizer.tokenize(self.queries_d[qid])
                def relevant(pid):
                    return pid in qrel_per_qid and qrel_per_qid[pid]

                def encode(text):
                    doc_tokens = self.tokenizer.tokenize(text)
                    averager.append(len(doc_tokens))

                pos_docs = list([(pid, content) for pid, content in passages if relevant(pid)])
                neg_docs = list([(pid, content) for pid, content in passages if not relevant(pid)])
                if not neg_docs:
                    print(qid)
                    print("len(pos_doc)", len(pos_docs))
                    print("len(neg_doc)", len(neg_docs))
                    raise IndexError
                for pos_pid, pos_content in pos_docs:
                    neg_pid, neg_content = pick1(neg_docs)
                    encode(pos_content)
                    encode(neg_content)
                    success_docs += 1
            except KeyError:
                missing_cnt += 1
                missing_doc_qid.append(qid)
        print('avg token length: ', averager.get_average())
        return []

    def write(self, insts: List[PairedInstance], out_path: str):
        for e in insts:
            pass
