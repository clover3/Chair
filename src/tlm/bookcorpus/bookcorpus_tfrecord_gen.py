import os
import random
from typing import List

from data_generator.job_runner import WorkerInterface, JobRunner, sydney_working_dir
from misc_lib import pick1
from tlm.bookcorpus.load_tokens import load_seg_with_repeat
from tlm.data_gen.base import truncate_seq_pair, format_tokens_pair_n_segid
from tlm.data_gen.lm_datagen import UnmaskedPairGen, SegmentInstance
from tlm.data_gen.run_sero_gen import pool_tokens, Token


class BookcorpusPairGen(UnmaskedPairGen):
    def __init__(self):
        super(UnmaskedPairGen, self).__init__()

    def create_instances_from_document(self, document):
        raise Exception()

    def create_instances_from_sent_list(self, sent_list):
        max_num_tokens = self.max_seq_length - 3
        target_seq_length = max_num_tokens
        print("pooling chunks")
        chunks: List[List[Token]] = pool_tokens(self.rng, sent_list, target_seq_length)

        target_inst_num = len(chunks)
        instances = []
        for _ in range(target_inst_num):
            chunk_1: List[Token] = pick1(chunks)
            if len(chunk_1) < 3:
                continue

            m = self.rng.randint(1, len(chunk_1))
            tokens_a = chunk_1[:m]
            b_length = target_seq_length - len(tokens_a)
            tokens_b = chunk_1[m:][:b_length]
            truncate_seq_pair(tokens_a, tokens_b, target_seq_length, self.rng)

            tokens, segment_ids = format_tokens_pair_n_segid(tokens_a, tokens_b)
            instance = SegmentInstance(
                tokens=tokens,
                segment_ids=segment_ids)
            instances.append(instance)

        return instances

class Worker(WorkerInterface):
    def __init__(self, out_path):
        self.out_dir = out_path
        self.gen = BookcorpusPairGen()

    def work(self, job_id):
        sent_list = load_seg_with_repeat(job_id)
        output_file = os.path.join(self.out_dir, "{}".format(job_id))
        insts = self.gen.create_instances_from_sent_list(sent_list)
        random.shuffle(insts)
        self.gen.write_instance_to_example_files(insts, [output_file])


if __name__ == "__main__":
    print("process started")
    runner = JobRunner(sydney_working_dir, 75 * 3, "bookcorpus_pair_x3", Worker)
    runner.start()

