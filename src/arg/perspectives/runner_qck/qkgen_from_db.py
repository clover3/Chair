import json
import os
from typing import NamedTuple, List, Dict, Iterable

from arg.perspectives.paraphrase.doc_classification_gen import enum_passages
from data_generator.job_runner import WorkerInterface
from data_generator.tokenizer_wo_tf import get_tokenizer
from datastore.interface import load_multiple
from datastore.table_names import BertTokenizedCluewebDoc
from galagos.parse import load_galago_ranked_list
from galagos.types import GalagoDocRankEntry
from misc_lib import exist_or_mkdir, DataIDManager, tprint
from tf_util.record_writer_wrap import write_records_w_encode_fn
from tlm.data_gen.base import get_basic_input_feature
from tlm.data_gen.bert_data_gen import create_int_feature


class Instance(NamedTuple):
    tokens1: List[str]
    tokens2: List[str]
    label: int
    data_id: int


class QKGenFromDB(WorkerInterface):
    def __init__(self, q_res_path, query_d: Dict[int, str], out_dir):
        self.ranked_list: Dict[str, List[GalagoDocRankEntry]] = load_galago_ranked_list(q_res_path)
        query_ids = list(self.ranked_list.keys())
        query_ids.sort()
        self.job_id_to_q_id = {job_id: q_id for job_id, q_id in enumerate(query_ids)}
        self.query_d: Dict[int, str] = query_d
        self.tokenizer = get_tokenizer()
        self.max_seq_length = 512
        self.out_dir = out_dir
        self.info_out_dir = out_dir + "_info"
        exist_or_mkdir(self.info_out_dir)

    def work(self, job_id):
        data_id_man = DataIDManager()
        insts = self.generate_instances(job_id, data_id_man)
        save_path = os.path.join(self.out_dir, str(job_id))

        def encode_fn(inst: Instance):
            tokens1 = inst.tokens1
            max_seg2_len = self.max_seq_length - 3 - len(tokens1)

            tokens2 = inst.tokens2[:max_seg2_len]
            tokens = ["[CLS]"] + tokens1 + ["[SEP]"] + tokens2 + ["[SEP]"]

            segment_ids = [0] * (len(tokens1) + 2) \
                          + [1] * (len(tokens2) + 1)
            tokens = tokens[:self.max_seq_length]
            segment_ids = segment_ids[:self.max_seq_length]
            features = get_basic_input_feature(self.tokenizer, self.max_seq_length, tokens, segment_ids)
            features['label_ids'] = create_int_feature([inst.label])
            features['data_id'] = create_int_feature([inst.data_id])
            return features

        write_records_w_encode_fn(save_path, encode_fn, insts)
        info_save_path = os.path.join(self.info_out_dir, str(job_id))
        json.dump(data_id_man.id_to_info, open(info_save_path, "w"))

    def generate_instances(self, job_id, data_id_man):
        q_id = self.job_id_to_q_id[job_id]
        query_text = self.query_d[int(q_id)]
        query_tokens = self.tokenizer.tokenize(query_text)
        ranked_list = self.ranked_list[q_id][:1000]
        doc_ids = list([e.doc_id for e in ranked_list])
        tprint("Loading documents start")
        docs_d: Dict[str, List[List[str]]] = load_multiple(BertTokenizedCluewebDoc, doc_ids, True)
        tprint("Loading documents done")
        avail_seq_length = self.max_seq_length - len(query_tokens) - 3

        label_dummy = 0
        not_found = 0
        for doc_id in doc_ids:
            try:
                doc: List[List[str]] = docs_d[doc_id]
                passages: Iterable[List[str]] = enum_passages(doc, avail_seq_length)

                for passage_idx, p in enumerate(passages):
                    if passage_idx > 9:
                        break
                    data_id = data_id_man.assign({
                        'query_id': q_id,
                        'doc_id': doc_id,
                        'passage_idx': passage_idx
                    })
                    yield Instance(query_tokens, p, label_dummy, data_id)
            except KeyError:
                not_found += 1
        print("{} of {} docs not found".format(not_found, len(doc_ids)))