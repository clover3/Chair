import json
import os
import pickle
from typing import List
from typing import NamedTuple

from arg.qck.decl import QCKQuery, QCKCandidate, get_light_qckquery, get_light_qckcandidate
from data_generator.create_feature import create_int_feature
from data_generator.data_parser.robust import load_robust_04_query
from data_generator.data_parser.robust2 import load_bm25_best
from data_generator.job_runner import sydney_working_dir
from data_generator.tokenizer_wo_tf import get_tokenizer
from misc_lib import DataIDManager, exist_or_mkdir
from tf_util.record_writer_wrap import RecordWriterWrap
from tlm.data_gen.base import get_basic_input_feature
from tlm.robust.load import robust_query_intervals


class Instance(NamedTuple):
    tokens: List[str]
    seg_ids: List[int]
    data_id: int
    label: int


class RobustPredictGen:
    def __init__(self, encoder, max_seq_length, top_k=100, query_type="title"):
        self.data = self.load_tokens_from_pickles()
        self.max_seq_length = max_seq_length
        self.queries = load_robust_04_query(query_type)
        self.galago_rank = load_bm25_best()
        self.top_k = top_k
        self.encoder = encoder
        self.tokenizer = get_tokenizer()

    @staticmethod
    def load_tokens_from_pickles():
        path = os.path.join(sydney_working_dir, "RobustPredictTokens3", "1")
        return pickle.load(open(path, "rb"))

    def generate(self, query_list, data_id_manager):
        all_insts = []
        for query_id in query_list:
            if query_id not in self.galago_rank:
                continue
            query = self.queries[query_id]
            qck_query = QCKQuery(query_id, "")
            query_tokens = self.tokenizer.tokenize(query)
            for doc_id, _, _ in self.galago_rank[query_id][:self.top_k]:
                tokens = self.data[doc_id]
                passage_list = self.encoder.encode(query_tokens, tokens)
                candidate = QCKCandidate(doc_id, "")
                for idx, (tokens, seg_ids) in enumerate(passage_list):
                    info = {
                        'query': get_light_qckquery(qck_query),
                        'candidate': get_light_qckcandidate(candidate),
                        'idx': idx
                    }
                    data_id = data_id_manager.assign(info)
                    inst = Instance(tokens, seg_ids, data_id, 0)
                    all_insts.append(inst)
        return all_insts

    def write(self, insts: List[Instance], out_path):
        writer = RecordWriterWrap(out_path)
        for inst in insts:
            feature = get_basic_input_feature(self.tokenizer, self.max_seq_length,
                                              inst.tokens,
                                              inst.seg_ids)
            feature["data_id"] = create_int_feature([int(inst.data_id)])
            feature["label_ids"] = create_int_feature([int(inst.label)])
            writer.write_feature(feature)

        writer.close()


class RobustPredictGenPrecise(RobustPredictGen):
    def generate(self, query_list, data_id_manager):
        all_insts = []
        for query_id in query_list:
            if query_id not in self.galago_rank:
                continue
            query = self.queries[query_id]
            qck_query = QCKQuery(query_id, "")
            query_tokens = self.tokenizer.tokenize(query)
            for doc_id, _, _ in self.galago_rank[query_id][:self.top_k]:
                tokens = self.data[doc_id]
                passage_list = self.encoder.encode(query_tokens, tokens)
                candidate = QCKCandidate(doc_id, "")
                for idx, (st, ed, tokens, seg_ids) in enumerate(passage_list):
                    info = {
                        'query': get_light_qckquery(qck_query),
                        'candidate': get_light_qckcandidate(candidate),
                        'idx': idx,
                        'st': st,
                        'ed': ed,
                    }
                    data_id = data_id_manager.assign(info)
                    inst = Instance(tokens, seg_ids, data_id, 0)
                    all_insts.append(inst)
        return all_insts


class RobustWorker:
    def __init__(self, generator, out_path):
        self.out_path = out_path
        self.gen = generator

    def work(self, job_id):
        st, ed = robust_query_intervals[job_id]
        out_path = os.path.join(self.out_path, str(st))
        max_inst_per_job = 1000 * 10000
        base = job_id * max_inst_per_job
        data_id_manager = DataIDManager(base, max_inst_per_job)
        query_list = [str(i) for i in range(st, ed+1)]
        insts = self.gen.generate(query_list, data_id_manager)
        self.gen.write(insts, out_path)

        info_dir = self.out_path + "_info"
        exist_or_mkdir(info_dir)
        info_path = os.path.join(info_dir, str(job_id) + ".info")
        json.dump(data_id_manager.id_to_info, open(info_path, "w"))

