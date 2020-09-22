import abc
import json
import os
from collections import OrderedDict
from typing import List, Iterable, Tuple, Any

from arg.qck.decl import QCKQuery, KDP, QKUnit
from data_generator.job_runner import WorkerInterface
from misc_lib import DataIDManager, exist_or_mkdir
from tf_util.record_writer_wrap import write_records_w_encode_fn


class InstanceGenerator(abc.ABC):
    @abc.abstractmethod
    def generate(self, kc_candidate: Iterable[QKUnit],
                       data_id_manager: DataIDManager):
        pass

    def encode_fn(self, any: Any) -> OrderedDict:
        pass


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
        base = job_id * 10000
        data_id_manager = DataIDManager(base)
        todo = self.qk_candidate[job_id:job_id + 1]
        insts: List = self.generator.generate(todo, data_id_manager)
        print("{} instances".format(len(insts)))
        save_path = os.path.join(self.out_dir, str(job_id))
        write_records_w_encode_fn(save_path, self.generator.encode_fn, insts)

        info_dir = self.out_dir + "_info"
        exist_or_mkdir(info_dir)
        info_path = os.path.join(info_dir, str(job_id) + ".info")
        json.dump(data_id_manager.id_to_info, open(info_path, "w"))


