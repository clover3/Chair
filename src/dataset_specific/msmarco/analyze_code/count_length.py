import os
import pickle
from collections import Counter

from data_generator.job_runner import WorkerInterface, JobRunner
from data_generator.tokenizer_wo_tf import get_tokenizer
from dataset_specific.msmarco.common import at_working_dir
from epath import job_man_dir
from misc_lib import TEL, exist_or_mkdir

MB = 1000 * 1000
MB = 1000 * 1000

def count_length(start_offset, end_offset):
    doc_f = open(at_working_dir("msmarco-docs.tsv"), encoding="utf8")
    tokenizer = get_tokenizer()
    doc_f.seek(start_offset)
    line = doc_f.readline()
    counter = Counter()
    cnt_in_mb = (end_offset - start_offset) / MB
    last_mb = start_offset

    print("{} MB".format(cnt_in_mb))
    ##
    bodies = []
    while doc_f.tell() < end_offset:
        line = doc_f.readline()
        doc_id, url, title, body = line.split("\t")
        bodies.append(body)

    print("{} documents".format(len(bodies)))

    for body in TEL(bodies):
        doc_len = len(tokenizer.tokenize(body))
        counter[doc_len] += 1
    return counter


job_size = 100 * MB

class Worker(WorkerInterface):
    def __init__(self, out_dir):
        self.out_dir = out_dir
        exist_or_mkdir(out_dir)

    def work(self, job_id):
        save_path = os.path.join(self.out_dir, str(job_id))
        st = job_size * job_id
        ed = st + job_size
        c = count_length(st, ed)
        pickle.dump(c, open(save_path, "wb"))


def main():
    # num_jobs = 229
    num_jobs = 40
    runner = JobRunner(job_man_dir, num_jobs, "MMD_doc_len_cnt", Worker)
    runner.start()


if __name__ == "__main__":
    main()