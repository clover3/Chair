import json
import os
from typing import List, Tuple

from arg.qck.decl import QCKQuery, KDP
from arg.qck.instance_generator.base import InstanceGenerator
from data_generator.job_runner import WorkerInterface
from misc_lib import DataIDManager, exist_or_mkdir, tprint
from tf_util.record_writer_wrap import write_records_w_encode_fn


class QCKWorker(WorkerInterface):
    def __init__(self,
                 qk_candidate: List[Tuple[QCKQuery, List[KDP]]],
                 instance_generator: InstanceGenerator,
                 out_dir):
        self.generator = instance_generator
        print("Total of {} jobs".format(len(qk_candidate)))
        self.out_dir = out_dir
        self.qk_candidate = qk_candidate

    def work(self, job_id):
        max_data_per_job = 1000 * 1000
        base = job_id * max_data_per_job
        data_id_manager = DataIDManager(base, base + max_data_per_job)
        todo = self.qk_candidate[job_id:job_id + 1]
        tprint("Generating instances")
        insts: List = self.generator.generate(todo, data_id_manager)
        tprint("{} instances".format(len(insts)))
        save_path = os.path.join(self.out_dir, str(job_id))
        tprint("Writing")
        write_records_w_encode_fn(save_path, self.generator.encode_fn, insts)
        tprint("writing done")

        info_dir = self.out_dir + "_info"
        exist_or_mkdir(info_dir)
        info_path = os.path.join(info_dir, str(job_id) + ".info")
        json.dump(data_id_manager.id_to_info, open(info_path, "w"))


class QCKWorkerWithNeg(WorkerInterface):
    def __init__(self,
                 qk_candidate: List[Tuple[QCKQuery, List[KDP]]],
                 instance_generator,
                 out_dir):
        self.generator = instance_generator
        print("Total of {} jobs".format(len(qk_candidate)))
        self.out_dir = out_dir
        self.qk_candidate = qk_candidate

    def work(self, job_id):
        max_data_per_job = 1000 * 1000
        base = job_id * max_data_per_job
        data_id_manager = DataIDManager(base, base + max_data_per_job)
        todo = self.qk_candidate[job_id:job_id + 1]
        other_qk = self.qk_candidate[:job_id] + self.qk_candidate[job_id+1:]
        tprint("Generating instances")
        insts: List = self.generator.generate(todo, other_qk, data_id_manager)
        tprint("{} instances".format(len(insts)))
        save_path = os.path.join(self.out_dir, str(job_id))
        tprint("Writing")
        write_records_w_encode_fn(save_path, self.generator.encode_fn, insts)
        tprint("writing done")

        info_dir = self.out_dir + "_info"
        exist_or_mkdir(info_dir)
        info_path = os.path.join(info_dir, str(job_id) + ".info")
        json.dump(data_id_manager.id_to_info, open(info_path, "w"))



class QCKWorkerMultiple(WorkerInterface):
    def __init__(self,
                 qk_candidate: List[Tuple[QCKQuery, List[KDP]]],
                 instance_generator: InstanceGenerator,
                 n_qk_per_job,
                 out_dir):
        self.generator = instance_generator
        print("Total of {} ".format(len(qk_candidate)))
        self.out_dir = out_dir
        self.qk_candidate = qk_candidate
        self.n_qk_per_job = n_qk_per_job

    def work(self, job_id):
        max_data_per_job = 1000 * 1000
        base = job_id * max_data_per_job
        data_id_manager = DataIDManager(base, base + max_data_per_job)
        start = job_id * self.n_qk_per_job
        end = start + self.n_qk_per_job
        todo = self.qk_candidate[start:end]
        tprint("Generating instances")
        insts: List = self.generator.generate(todo, data_id_manager)
        tprint("{} instances".format(len(insts)))
        save_path = os.path.join(self.out_dir, str(job_id))
        tprint("Writing")
        write_records_w_encode_fn(save_path, self.generator.encode_fn, insts)
        tprint("writing done")

        info_dir = self.out_dir + "_info"
        exist_or_mkdir(info_dir)
        info_path = os.path.join(info_dir, str(job_id) + ".info")
        json.dump(data_id_manager.id_to_info, open(info_path, "w"))



