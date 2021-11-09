import json
import os

from data_generator.job_runner import WorkerInterface
from misc_lib import exist_or_mkdir, DataIDManager, tprint
from tlm.data_gen.msmarco_doc_gen.mmd_gen_common import MMDGenI


class MMDWorker(WorkerInterface):
    def __init__(self, query_group, generator: MMDGenI, out_dir):
        self.out_dir = out_dir
        self.query_group = query_group
        self.generator: MMDGenI = generator
        self.info_dir = os.path.join(self.out_dir + "_info")
        exist_or_mkdir(self.info_dir)

    def work(self, job_id):
        qids = self.query_group[job_id]
        data_bin = 1000000
        data_id_st = job_id * data_bin
        data_id_ed = data_id_st + data_bin
        data_id_manager = DataIDManager(data_id_st, data_id_ed)
        tprint("generating instances")
        insts = self.generator.generate(data_id_manager, qids)
        # tprint("{} instances".format(len(insts)))
        out_path = os.path.join(self.out_dir, str(job_id))
        self.generator.write(insts, out_path)

        info_path = os.path.join(self.info_dir, "{}.info".format(job_id))
        json.dump(data_id_manager.id_to_info, open(info_path, "w"))