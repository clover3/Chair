import os
from functools import partial

from data_generator.job_runner import JobRunner, sydney_working_dir
from tlm.data_gen.adhoc_datagen import RobustTrainGen, FirstSegmentAsDoc


# There is four-level hierarchy for generating data for robust
# 1. JobRunner : this is basic job runner
# 2. Worker : RobustWorker  -> give range of queries to generator
# 3. Generator : RobustTrainGen, RobustPredictGen : Whether to make instance paired or not
# 4. Encoder : How the each query/document pair is encoded

class RobustWorker:
    def __init__(self, generator, out_path):
        self.out_path = out_path
        self.gen = generator

    def work(self, job_id):
        intervals = [(301,350), (351,400), (401,450), (601,650), (651, 700)]
        st, ed = intervals[job_id]
        out_path = os.path.join(self.out_path, str(st))

        query_list = [str(i) for i in range(st, ed+1)]
        insts = self.gen.generate(query_list)
        self.gen.write(insts, out_path)


def generate_robust_first_for_train():
    max_seq_length = 512
    encoder = FirstSegmentAsDoc(max_seq_length)
    worker_factory = partial(RobustWorker, RobustTrainGen(encoder, max_seq_length))
    runner = JobRunner(sydney_working_dir, 4, "RobustFirstClean", worker_factory)
    runner.start()


if __name__ == "__main__":
    generate_robust_first_for_train()

