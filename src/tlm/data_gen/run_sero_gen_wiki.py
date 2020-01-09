import os
import random
import time
from collections import Counter

from data_generator.job_runner import sydney_working_dir, JobRunner
from misc_lib import flatten, average
from tf_util.record_writer_wrap import RecordWriterWrap
from tf_util.tf_logging import tf_logging
from tlm.data_gen.base import LMTrainGen, get_basic_input_feature
from tlm.data_gen.bert_data_gen import create_int_feature


class WikiWorker:
    def __init__(self, out_path):
        self.out_dir = out_path
        target_seq_len = (128 - 2) * 24
        self.gen = WikiGen(target_seq_len, True)

    def work(self, job_id):
        doc_id = job_id
        if doc_id > 1000:
            doc_id = doc_id % 1000

        docs = self.gen.load_doc_seg(doc_id)
        output_file = os.path.join(self.out_dir, "{}".format(job_id))
        inst_list = []
        for doc in docs:
            insts = self.gen.create_instances_from_document(doc)
            inst_list.extend(insts)

        avg_len = average([len(t[0]) for t in inst_list])
        tf_logging.info("{} docs, {} chunks, avg_Len={}".format(len(docs), len(inst_list), avg_len))

        histogram = Counter()
        for inst in inst_list:
            tokens, _ = inst
            num_window = int(len(tokens) / 126)
            histogram[num_window] += 1

        random.shuffle(inst_list)
        self.gen.write_instances(inst_list, output_file)


class WikiGen(LMTrainGen):
    def __init__(self, target_seq_length, drop_short=True):
        super(WikiGen, self).__init__()
        self.rng = random.Random(time.time())
        self.target_seq_length = target_seq_length
        self.drop_short = drop_short

    def pool_tokens(self, sent_list, target_seq_length, skip=False):
        results = []
        current_chunk = []
        current_length = 0
        i = 0
        if skip:
            i = i + self.rng.randint(0, 3)

        while i < len(sent_list):
            segment = sent_list[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(sent_list) - 1 or current_length >= target_seq_length:
                tokens_a = flatten(current_chunk)
                tokens_a = tokens_a[:target_seq_length]
                results.append(tokens_a)
                current_chunk = []
                current_length = 0
                if skip:
                    i = i + self.rng.randint(0, 3)
            i += 1
        return results

    def create_instances_from_document(self, doc):
        length_l = []
        for tokens in self.pool_tokens(doc, self.target_seq_length, True):
            l = len(tokens)
            if self.drop_short and l < 0.5 * self.target_seq_length:
                continue
            length_l.append(l)
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


if __name__ == "__main__":
    working_dir = sydney_working_dir
    runner = JobRunner(working_dir, 1000, "sero_wiki_2", WikiWorker)
    runner.start()
