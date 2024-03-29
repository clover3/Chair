import json
import os
import pickle
from typing import List, Dict

from data_generator.job_runner import WorkerInterface
from list_lib import lflatten
from misc_lib import exist_or_mkdir, DataIDManager, tprint
from tlm.data_gen.msmarco_doc_gen.gen_worker_qd import QDDistillGenI


class DistillGenWorker(WorkerInterface):
    def __init__(self, generator: QDDistillGenI,
                 query_group,
                 resource_path_format, out_dir):
        self.out_dir = out_dir
        self.generator: QDDistillGenI = generator
        self.info_dir = os.path.join(self.out_dir + "_info")
        self.resource_path_format = resource_path_format
        exist_or_mkdir(self.info_dir)
        self.query_group = query_group

    def work(self, job_id):
        print("work", job_id)
        qids = self.query_group[job_id]
        data_bin = 1000000
        data_id_st = job_id * data_bin
        data_id_ed = data_id_st + data_bin
        data_id_manager = DataIDManager(data_id_st, data_id_ed)
        parsed_data_path = self.resource_path_format.format(job_id)

        if not os.path.exists(parsed_data_path):
            return
        tprint("Loading pickle from {}".format(parsed_data_path))
        entries: List[Dict] = pickle.load(open(parsed_data_path, "rb"))
        tprint("generating instances")
        insts = self.generator.generate(data_id_manager, qids, entries)
        out_path = os.path.join(self.out_dir, str(job_id))
        self.generator.write(insts, out_path)

        info_path = os.path.join(self.info_dir, "{}.info".format(job_id))
        json.dump(data_id_manager.id_to_info, open(info_path, "w"))


class DistillGenWorker10Grouped(WorkerInterface):
    def __init__(self, generator: QDDistillGenI,
                 query_group,
                 resource_path_format, out_dir):
        self.out_dir = out_dir
        self.generator: QDDistillGenI = generator
        self.info_dir = os.path.join(self.out_dir + "_info")
        self.resource_path_format = resource_path_format
        exist_or_mkdir(self.info_dir)
        self.query_group = query_group

    def work(self, job_id):
        print("work", job_id)
        st = job_id * 10
        ed = (job_id+1) * 10
        qids = lflatten(self.query_group[st:ed])
        data_bin = 1000000
        data_id_st = job_id * data_bin
        data_id_ed = data_id_st + data_bin
        data_id_manager = DataIDManager(data_id_st, data_id_ed)
        parsed_data_path = self.resource_path_format.format(job_id)

        if not os.path.exists(parsed_data_path):
            return
        tprint("Loading pickle from {}".format(parsed_data_path))
        entries: List[Dict] = pickle.load(open(parsed_data_path, "rb"))
        tprint("generating instances")
        insts = self.generator.generate(data_id_manager, qids, entries)
        out_path = os.path.join(self.out_dir, str(job_id))
        self.generator.write(insts, out_path)

        info_path = os.path.join(self.info_dir, "{}.info".format(job_id))
        json.dump(data_id_manager.id_to_info, open(info_path, "w"))