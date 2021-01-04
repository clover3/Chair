import os
import subprocess

from cpath import output_path
from data_generator.job_runner import WorkerInterface, JobRunner
from epath import job_man_dir
from galagos.parse import load_queries


class GetRMTermWorker(WorkerInterface):
    def __init__(self, queries, out_dir):
        self.index_path = "/mnt/lustre/godzilla/harding/ClueWeb12/idx/ClueWeb12-Disk1_00.idx"
        self.out_dir = out_dir
        self.queries = queries

    def work(self, job_id):
        q = self.queries[job_id]
        out_path = os.path.join(self.out_dir, q['number'])
        cmd_format = "galago get-rm-terms --index={} --numTerms=500  --requested=10 --query=\"{}\""
        cmd = cmd_format.format(self.index_path, q['text'], out_path)
        fout = open(out_path, "wb")
        p = subprocess.Popen(cmd, shell=True, stdout=fout)
        p.wait()
        fout.close()



if __name__ == "__main__":
    split = "train"
    split = "dev"
    query_path = os.path.join(output_path, "perspective_{}_claim_query_k0_fixed.json".format(split))
    queries = load_queries(query_path)
    num_jobs = len(queries) - 1
    job_name = "pc_rm_terms_{}".format(split)

 #
    def worker_factory(out_dir):
        return GetRMTermWorker(queries, out_dir)

    runner = JobRunner(job_man_dir, num_jobs, job_name, worker_factory)
    runner.start()

