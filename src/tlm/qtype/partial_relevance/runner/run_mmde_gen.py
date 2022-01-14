import json
import os
from typing import List, Dict

from cache import load_json_cache
from data_generator.job_runner import JobRunner, WorkerInterface
from dataset_specific.msmarco.common import QueryID
from epath import job_man_dir
from misc_lib import DataIDManager, exist_or_mkdir, tprint
from tlm.data_gen.adhoc_sent_tokenize import FromTextEncoder
from tlm.data_gen.msmarco_doc_gen.mmd_gen_common import MMDGenI
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResourcePredict
from tlm.qtype.partial_relevance.mmde_gen import MMDEPredictionGen


class MMDEWorker(WorkerInterface):
    def __init__(self, query_group, generator: MMDGenI, out_dir):
        self.out_dir = out_dir
        self.query_group = query_group
        self.generator: MMDGenI = generator
        self.info_dir = os.path.join(self.out_dir + "_info")
        exist_or_mkdir(self.info_dir)

    def work(self, job_id):
        qids = self.query_group[str(job_id)]
        data_bin = 1000000
        data_id_st = job_id * data_bin
        data_id_ed = data_id_st + data_bin
        data_id_manager = DataIDManager(data_id_st, data_id_ed)
        tprint("generating instances")
        insts = self.generator.generate(data_id_manager, qids)
        out_path = os.path.join(self.out_dir, str(job_id))
        self.generator.write(insts, out_path)

        info_path = os.path.join(self.info_dir, "{}.info".format(job_id))
        json.dump(data_id_manager.id_to_info, open(info_path, "w"))


if __name__ == "__main__":
    split = "dev"
    resource = ProcessedResourcePredict(split)
    max_seq_length = 512
    document_encoder = FromTextEncoder(max_seq_length, True, None, 20)
    generator = MMDEPredictionGen(resource, document_encoder, max_seq_length)
    selected_qids_grouped: Dict[str, List[QueryID]] = load_json_cache("MMD_selected_qids")

    def factory(out_dir):
        return MMDEWorker(selected_qids_grouped, generator, out_dir)

    runner = JobRunner(job_man_dir, len(resource.query_group)-1, "MMDE_{}".format(split), factory)
    runner.start()
