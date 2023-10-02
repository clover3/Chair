import os
import pickle
from collections import Counter

from adhoc.kn_tokenizer import KrovetzNLTKTokenizer
from cpath import output_path
from data_generator.job_runner import WorkerInterface
from table_lib import tsv_iter
from job_manager.job_runner_with_server import JobRunnerS
from misc_lib import path_join, TELI


class Worker(WorkerInterface):
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.tokenizer = KrovetzNLTKTokenizer()

    def work(self, job_no):
        save_path = os.path.join(self.out_dir, str(job_no))
        file_path = path_join(output_path, "msmarco", "passage", "when_full_re", str(job_no))
        itr = tsv_iter(file_path)
        output = []
        for item in TELI(itr, 1000*1000):
            qid, pid, query, text = item
            qtf = Counter(self.tokenizer.tokenize_stem(query))
            ptf = Counter(self.tokenizer.tokenize_stem(text))
            output.append((qid, pid, qtf, ptf))
        pickle.dump(output, open(save_path, "wb"))


def main():
    working_dir = path_join("output", "msmarco", "passage")
    runner = JobRunnerS(working_dir, 17, "when_full_re_tokenized", Worker)
    runner.start()



if __name__ == "__main__":
    main()