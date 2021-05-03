import os

from data_generator.tokenizer_wo_tf import get_tokenizer
from dataset_specific.msmarco.common import at_working_dir
from typing import List, Iterable, Callable, Dict, Tuple, Set

from dataset_specific.msmarco.corpus_tokenize.line_offset_test import load_offset
from epath import job_man_dir
from job_manager.job_runner_with_server import JobRunnerS


def tokenize_lines(tokenizer, lines):
    new_lines = []
    for line in lines:
        if not line.strip():
            break
        try:
            docid, url, title, body = line.split("\t")
            title_tokens = " ".join(tokenizer.tokenize(title))
            body_tokens = " ".join(tokenizer.tokenize(body))
            new_lines.append((docid, url, title_tokens, body_tokens))
        except ValueError:
            print("Exception")
            print(line)
            raise
            pass
    return new_lines


class CorpusTokenizeWorker:
    def __init__(self, out_dir):
        self.doc_f = open(at_working_dir("msmarco-docs.tsv"), encoding="utf8")
        self.line_offset_d: List[int] = load_offset()
        self.out_dir = out_dir
        self.tokenizer = get_tokenizer()

    def work(self, job_id):
        job_size = 1000
        line_start = job_id * job_size
        line_end = line_start + job_size

        self.doc_f.seek(self.line_offset_d[line_start])

        lines = []
        for i in range(job_size):
            line = self.doc_f.readline()
            lines.append(line)

        cur_offset = self.doc_f.tell()
        try:
            expected_offset = self.line_offset_d[line_end]
            if cur_offset != expected_offset:
                print("cur_offset != expected_offset : {} != {}".format(cur_offset, expected_offset))
        except IndexError as exception_e:
            print(exception_e)

        tokenized_lines = tokenize_lines(self.tokenizer, lines)
        out_f = open(os.path.join(self.out_dir, str(job_id)), "w")

        for row in tokenized_lines:
            out_f.write("\t".join(row) + "\n")


if __name__ == "__main__":
    num_job = 3213+1
    runner = JobRunnerS(job_man_dir, num_job, "MSMARCO_tokens", CorpusTokenizeWorker)
    runner.start()

