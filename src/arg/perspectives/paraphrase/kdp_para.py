import json
import os
from collections import OrderedDict
from typing import List, Dict, NamedTuple

# 100 claim * 5 p_cluster * 30 sentences * 10 docs  = 150000 instance
from arg.perspectives.evaluate import perspective_getter
from arg.perspectives.load import get_claim_perspective_id_dict
from data_generator.create_feature import create_int_feature
from data_generator.job_runner import WorkerInterface, JobRunner
from data_generator.tokenizer_wo_tf import get_tokenizer
from datastore.interface import preload_man, load
from datastore.table_names import BertTokenizedCluewebDoc
from epath import job_man_dir
from exec_lib import run_func_with_config
from galagos.parse import load_galago_ranked_list
from galagos.types import SimpleRankedListEntry
from list_lib import lmap, dict_value_map
from misc_lib import DataIDManager, exist_or_mkdir
from tf_util.record_writer_wrap import write_records_w_encode_fn
from tlm.data_gen.base import get_basic_input_feature


# retrieve top relevant documents
# To save cost, check first 30 sentences


def first_pid_as_rep() -> Dict[int, List[int]]:
    id_dict: Dict[int, List[List[int]]] = get_claim_perspective_id_dict()
    id_dict_small: Dict[int, List[int]] = dict_value_map(lambda ll: lmap(lambda l: l[0], ll), id_dict)
    return id_dict_small


class Instance(NamedTuple):
    pid: int
    sent: List[str]
    data_id: int


class Writer:
    def __init__(self, max_seq_length, reverse=False):
        self.p_tokens_d: Dict[int, List[str]] = {}
        self.tokenizer = get_tokenizer()
        self.max_seq_length = max_seq_length
        self.reverse = reverse

    def get_p_tokens(self, pid: int):
        if pid not in self.p_tokens_d:
            text = perspective_getter(pid)
            self.p_tokens_d[pid] = self.tokenizer.tokenize(text)
        return self.p_tokens_d[pid]

    def encode(self, inst: Instance) -> OrderedDict:
        if not self.reverse:
            tokens1 = self.get_p_tokens(inst.pid)
            tokens2 = inst.sent
        else:
            tokens1 = inst.sent
            tokens2 = self.get_p_tokens(inst.pid)
        tokens = ["[CLS]"] + tokens1 + ["[SEP]"] + tokens2 + ["[SEP]"]
        segment_ids = [0] * (len(tokens1) + 2) \
                      + [1] * (len(tokens2) + 1)
        max_seq_length = self.max_seq_length
        tokens = tokens[:max_seq_length]
        segment_ids = segment_ids[:max_seq_length]
        features = get_basic_input_feature(self.tokenizer, max_seq_length, tokens, segment_ids)
        features['label_ids'] = create_int_feature([0])
        features['data_ids'] = create_int_feature([inst.data_id])
        return features


class KDPParaWorker(WorkerInterface):
    def __init__(self, config, writer, out_dir, ):
        q_res_path = config['q_res_path']
        self.top_n = config['top_n']
        self.num_sent = config['num_sent']
        self.max_seq_length = config['max_seq_length']
        self.ranked_list: Dict[str, List[SimpleRankedListEntry]] = load_galago_ranked_list(q_res_path)
        self.cids = lmap(int, self.ranked_list.keys())
        self.pid_dict = first_pid_as_rep()
        self.out_dir = out_dir
        self.writer = writer

    def work(self, job_id):
        cid = self.cids[job_id]
        entries: List[SimpleRankedListEntry] = self.ranked_list[str(cid)]
        max_items = 1000 * 1000
        base = job_id * max_items
        end = base + max_items
        data_id_manager = DataIDManager(base, end)
        insts = self.get_instances(cid, data_id_manager, entries)
        save_path = os.path.join(self.out_dir, str(job_id))
        writer = self.writer
        write_records_w_encode_fn(save_path, writer.encode, insts)
        info_dir = self.out_dir + "_info"
        exist_or_mkdir(info_dir)
        info_path = os.path.join(info_dir, str(job_id) + ".info")
        json.dump(data_id_manager.id_to_info, open(info_path, "w"))

    def get_instances(self, cid, data_id_manager, entries):
        doc_ids = lmap(lambda x: x.doc_id, entries)
        preload_man.preload(BertTokenizedCluewebDoc, doc_ids)
        n_doc_not_found = 0
        for entry in entries[:self.top_n]:
            try:
                tokens: List[List[str]] = load(BertTokenizedCluewebDoc, entry.doc_id)
                for sent_idx, sent in enumerate(tokens[:self.num_sent]):
                    for pid in self.pid_dict[int(cid)]:
                        info = {'cid': cid,
                                'pid': pid,
                                'doc_id': entry.doc_id,
                                'sent_idx': sent_idx
                                }
                        yield Instance(pid, sent, data_id_manager.assign(info))
            except KeyError:
                n_doc_not_found += 1
        if n_doc_not_found:
            print("{} of {} docs not found".format(n_doc_not_found, len(doc_ids)))
    # print


def main(config):
    def get_worker(out_dir):
        writer = Writer(max_seq_length=config['max_seq_length'], reverse=config['reverse'])
        return KDPParaWorker(config, writer, out_dir)

    q_res_path = config['q_res_path']
    ranked_list: Dict[str, List[SimpleRankedListEntry]] = load_galago_ranked_list(q_res_path)
    num_job = len(ranked_list)-1

    runner = JobRunner(job_man_dir, num_job, config['job_name'], get_worker)
    runner.auto_runner()


if __name__ == "__main__":
    run_func_with_config(main)