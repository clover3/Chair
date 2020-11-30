import os
import pickle

from data_generator.adhoc.robust_tokenizer import RobustPreprocessTrain
from data_generator.job_runner import JobRunner, sydney_working_dir
from tlm.robust.load import robust_chunk_num


class TrainTokenWorker:
    def __init__(self, out_path):
        self.out_path = out_path
        self.gen = RobustPreprocessTrain()

    def work(self, job_id):
        out_path = os.path.join(self.out_path, str(job_id))
        token_d = self.gen.tokenize(job_id)
        pickle.dump(token_d, open(out_path, "wb"))


if __name__ == "__main__":
    runner = JobRunner(sydney_working_dir, robust_chunk_num, "RobustTokensClean", TrainTokenWorker)
    runner.start()


