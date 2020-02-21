import os
import pickle
import time

from cpath import data_path
from data_generator.job_runner import JobRunner, sydney_working_dir
from data_generator.tokenizer_wo_tf import FullTokenizer
from misc_lib import get_dir_files2


def load_undone_file_list():
    f = open("/mnt/nfs/work3/youngwookim/data/undone_list.txt", "r")
    l = []
    for line in f:
        l.append(line.strip())
    return l


class Worker:
    def __init__(self, out_path):
        self.out_dir = "/mnt/nfs/work3/youngwookim/data/clueweb12-B13_tokens"
        voca_path = os.path.join(data_path, "bert_voca.txt")
        self.tokenizer = FullTokenizer(voca_path, True)

        self.file_list = load_undone_file_list()

    def job_id_to_dir_path(self, job_id):
        if job_id < len(self.file_list):
            return self.file_list[job_id]
        else:
            assert False

    def work(self, job_id):
        if job_id >= len(self.file_list):
            return

        dir_path = self.job_id_to_dir_path(job_id)

        _, file_name = os.path.split(dir_path)
        output_name = self.input_name_to_output_name(file_name)
        output_path = os.path.join(self.out_dir, output_name)

        begin = time.time()
        self.do_tokenize(dir_path, output_path, self.tokenizer.tokenize)
        end = time.time()
        print("Elapsed: ", end-begin)

    @staticmethod
    def input_name_to_output_name(name):
        dropping_tail = ".warc.gz"
        return name[:-len(dropping_tail)]

    def do_tokenize(self, dir_path, outpath, tokenize_fn):
        #  Read jsonl
        token_data = {}

        raw_data = []
        for file_path in get_dir_files2(dir_path):
            _, file_name = os.path.split(file_path)
            _, dir_name = os.path.split(dir_path)

            doc_name = "{}_{}".format(file_name, dir_name)
            f = open(file_path, "r", errors="ignore")
            lines = f.readlines()
            raw_data.append((doc_name, lines))
            f.close()

        for doc_name, lines in raw_data:
            chunks = []
            try:
                for line in lines:
                    tokens = tokenize_fn(line)
                    chunks.append(tokens)
                token_data[doc_name] = chunks
            except FileNotFoundError:
                pass
            except Exception as e:
                print(doc_name)
                raise

        # save tokenized docs as pickles
        pickle.dump(token_data, open(outpath, "wb"))


if __name__ == "__main__":
    print("process started")
    runner = JobRunner(sydney_working_dir, 8000, "clueweb_tokenize", Worker)
    runner.start()


