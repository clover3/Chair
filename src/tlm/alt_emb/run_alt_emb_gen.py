import os
from typing import List

from data_generator.common import get_tokenizer
from data_generator.job_runner import JobRunner, sydney_working_dir
from tlm.alt_emb.add_alt_emb import convert_alt_emb


class Worker:
    def __init__(self, word_list: List[str], out_path):
        self.out_dir = out_path
        tokenizer = get_tokenizer()
        self.seq_set: List[List[int]] = []

        for word in word_list:
            subwords = tokenizer.tokenize(word)
            ids = tokenizer.convert_tokens_to_ids(subwords)
            print(subwords, ids)
            self.seq_set.append(ids)


    def work(self, job_id):
        input_path = os.path.join(sydney_working_dir, "unmasked_pair_x3", str(job_id))
        output_file = os.path.join(self.out_dir, "{}".format(job_id))
        convert_alt_emb(input_path, output_file, self.seq_set)


if __name__ == "__main__":
    def worker_gen(out_dir):
        word_list = ["heavy metal"]
        return Worker(word_list, out_dir)


    runner = JobRunner(sydney_working_dir, 4000, "alt_emb_heavy_metal", worker_gen)
    runner.start()



