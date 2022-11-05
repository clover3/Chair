import itertools
import os
from typing import List, Iterable

from cpath import at_output_dir
from data_generator.job_runner import WorkerInterface
from data_generator2.segmented_enc.es.evidence_candidate_gen import EvidenceCandidateGenerator
from dataset_specific.mnli.mnli_reader import MNLIReader
from epath import job_man_dir
from job_manager.job_runner_with_server import JobRunnerS
from misc_lib import exist_or_mkdir, ceil_divide
from tf_util.record_writer_wrap import write_records_w_encode_fn
from data_generator2.segmented_enc.es.common import PHSegmentedPair, get_ph_segment_pair_encode_fn


class Worker(WorkerInterface):
    def __init__(self, n_per_job, output_dir):
        exist_or_mkdir(output_dir)
        self.output_dir = output_dir
        self.num_candidate = 10
        self.ecg = EvidenceCandidateGenerator(self.num_candidate)
        segment_len = 300
        self.encode_fn = get_ph_segment_pair_encode_fn(segment_len)
        self.n_per_job = n_per_job

    def work(self, job_id):
        st = self.n_per_job * job_id
        ed = st + self.n_per_job
        reader = MNLIReader()
        src_itr = itertools.islice(reader.load_split("train"), st, ed)
        payload: Iterable[PHSegmentedPair] = self.ecg.generate(src_itr)
        num_out_item = (ed - st) * self.num_candidate
        output_path = os.path.join(self.output_dir, str(job_id))
        write_records_w_encode_fn(output_path, self.encode_fn, payload, num_out_item)


def main():
    split = "train"
    reader = MNLIReader()
    n_per_job = 10000
    n_jobs = ceil_divide(reader.get_data_size(split), n_per_job)

    def factory(out_dir):
        return Worker(n_per_job, out_dir)

    runner = JobRunnerS(job_man_dir, n_jobs, "evidence_candidate_gen", factory)
    runner.start()


if __name__ == "__main__":
    main()