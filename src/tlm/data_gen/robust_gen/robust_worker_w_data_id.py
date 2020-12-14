import json
import os

from misc_lib import DataIDManager, exist_or_mkdir
from tlm.robust.load import robust_query_intervals


class RobustWorkerWDataID:
    def __init__(self, generator, out_path):
        self.out_path = out_path
        self.info_dir = out_path + "_info"
        exist_or_mkdir(self.info_dir)
        self.gen = generator

    def work(self, job_id):
        st, ed = robust_query_intervals[job_id]
        out_path = os.path.join(self.out_path, str(st))

        query_list = [str(i) for i in range(st, ed+1)]
        unit = 1000 * 1000
        base = unit * job_id
        end = base + unit
        data_id_manager = DataIDManager(base, end)
        insts = self.gen.generate(data_id_manager, query_list)
        self.gen.write(insts, out_path)

        info_path = os.path.join(self.info_dir, str(st))
        json.dump(data_id_manager.id_to_info, open(info_path, "w"))


