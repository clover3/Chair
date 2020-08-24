import os
import random
import time
from typing import List, NewType, Any

from data_generator.job_runner import JobRunner, sydney_working_dir
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import flatten
from tf_util.record_writer_wrap import RecordWriterWrap
from tf_util.tf_logging import tf_logging
from tlm.bookcorpus.load_tokens import load_doc_seg
from tlm.data_gen.base import get_basic_input_feature
from tlm.data_gen.bert_data_gen import create_int_feature

Token = NewType('Token', Any)


def pool_tokens(rng, sent_list: List[List[Token]], target_seq_length, skip=False) -> List[List[Token]]:
    results: List[List[Token]] = []
    current_chunk = []
    current_length = 0
    i = 0
    if skip:
        i = i + rng.randint(0, 3)

    def is_new_doc(segment):
        return 'isbn' in segment

    num_real_doc = 1
    while i < len(sent_list):
        segment: List[Token] = sent_list[i]
        if is_new_doc(segment):
            num_real_doc += 1
            tokens_a: List[Token] = list(flatten(current_chunk))
            tokens_a = tokens_a[:target_seq_length]
            results.append(tokens_a)
            current_chunk: List[List[Token]] = []
            current_length = 0

        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(sent_list) - 1 or current_length >= target_seq_length:
            tokens_a = list(flatten(current_chunk))
            tokens_a = tokens_a[:target_seq_length]
            results.append(tokens_a)
            current_chunk = []
            current_length = 0
            if skip:
                i = i + rng.randint(0, 3)
        i += 1
    return results


class BookCorpusGen:
    def __init__(self, target_seq_length):
        self.tokenizer = get_tokenizer()
        self.rng = random.Random(time.time())
        self.target_seq_length = target_seq_length

    def pool_tokens(self, sent_list, target_seq_length, skip=False):
        return pool_tokens(self.rng, sent_list, target_seq_length, skip)


    def load_doc_seg(self, doc_id):
        return load_doc_seg(doc_id)

    def create_instances_from_text_piece(self, doc):
        for tokens in self.pool_tokens(doc, self.target_seq_length, True):
            l = len(tokens)
            segment_ids = [1] * l
            yield tokens, segment_ids

    def write_instances(self, new_inst_list, outfile):
        writer = RecordWriterWrap(outfile)
        example_numbers = []

        for (inst_index, instance) in enumerate(new_inst_list):
            tokens, segment_ids = instance
            features = get_basic_input_feature(self.tokenizer, self.target_seq_length, tokens, segment_ids)
            features["use_context"] = create_int_feature([1])
            writer.write_feature(features)
        writer.close()
        tf_logging.info("Wrote %d total instances", writer.total_written)
        return example_numbers


# 40 + 34 segments
class BookCorpusWorker:
    def __init__(self, out_path):
        self.out_dir = out_path
        target_seq_len = (128 - 2) * 128
        self.gen = BookCorpusGen(target_seq_len)

    def work(self, job_id):
        n_docs = 74
        split_factor = 1
        piece_id = int(job_id/split_factor)
        offset = job_id % split_factor
        if piece_id >= n_docs :
            piece_id = piece_id % n_docs
        text_piece = self.gen.load_doc_seg(piece_id)
        lines_per_doc = 1000 * 1000
        lines_per_job = int(lines_per_doc / split_factor)
        st = offset * lines_per_job
        ed = (offset + 1) * lines_per_job

        random_shift = random.randint(0, 1000) - 500
        st = max(0, st + random_shift)
        ed = min(lines_per_doc, ed + random_shift)

        output_file = os.path.join(self.out_dir, "{}".format(job_id))
        text_piece = text_piece[st:ed]
        insts = self.gen.create_instances_from_text_piece(text_piece)
        self.gen.write_instances(insts, output_file)


if __name__ == "__main__":
    working_dir = sydney_working_dir
    runner = JobRunner(working_dir, 74, "sero_book_corpus_gamma", BookCorpusWorker)
    runner.start()


