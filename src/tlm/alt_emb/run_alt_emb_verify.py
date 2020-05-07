import os
from typing import List

from base_type import FileName
from cpath import pjoin
from data_generator.job_runner import JobRunner, sydney_working_dir
from data_generator.tokenizer_wo_tf import get_tokenizer
from tlm.alt_emb.add_alt_emb import verify_alt_emb


class Worker:
    def __init__(self, word_list: List[str], out_path):
        self.out_dir = out_path
        tokenizer = get_tokenizer()
        self.seq_set: List[List[int]] = []
        self.input_dir = pjoin(sydney_working_dir, FileName("alt_emb_heavy_metal"))

        for word in word_list:
            subwords = tokenizer.tokenize(word)
            ids = tokenizer.convert_tokens_to_ids(subwords)
            print(subwords, ids)
            self.seq_set.append(ids)

    def work(self, job_id):
        input_path = os.path.join(self.input_dir, "{}".format(job_id))
        try:
            verify_alt_emb(input_path, self.seq_set)

        except Exception:
            pass


if __name__ == "__main__":
    def worker_gen(out_dir):
        word_list = ["heavy metal"]
        return Worker(word_list, out_dir)


    runner = JobRunner(sydney_working_dir, 4000, "alt_emb_heavy_metal_verify", worker_gen)
    runner.start()



