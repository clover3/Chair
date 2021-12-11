import os
import pickle

from data_generator.job_runner import WorkerInterface
from dataset_specific.msmarco.common import train_query_group_len
from epath import job_man_dir
from job_manager.job_runner_with_server import JobRunnerS


class Worker(WorkerInterface):
    def __init__(self, split, out_dir):
        self.split = split
        self.out_dir = out_dir

    def work(self, job_id):

        save_path = os.path.join(job_man_dir,
                                 "MSMARCO_{}_title_body_tokens_working".format(self.split),
                                 str(job_id))
        print(f"Job {job_id}: Reading file")
        f = open(save_path, "r")
        lines = f.readlines()
        f.close()
        print(f"Job {job_id}: Parsing file")
        tokens_d = {}
        for line in lines:
            docid, url, title_tokens_s, body_tokens_s = line.split("\t")
            tokens_d[docid] = title_tokens_s.split(), body_tokens_s.split()

        print(f"Job {job_id}: Saving file")
        save_path = os.path.join(self.out_dir, str(job_id))
        pickle.dump(tokens_d, open(save_path, "wb"))


def main():
    split = "train"
    def factory(out_dir):
        return Worker(split, out_dir)
    runner = JobRunnerS(job_man_dir, train_query_group_len, "MMD_{}_corpuswise_tokens".format(split), factory)
    runner.start()


if __name__ == "__main__":
    main()
