import os
import pickle
from collections import OrderedDict
from typing import List

from arg.perspectives.encode_paragraph_feature_to_tfrecord import format_feature
from arg.perspectives.select_paragraph import ParagraphClaimPersFeature
from data_generator.common import get_tokenizer
from data_generator.job_runner import JobRunner, sydney_working_dir
from list_lib import foreach
from tf_util.record_writer_wrap import RecordWriterWrap


class Worker:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.tokenizer = get_tokenizer()
        self.max_seq_length = 512
        self.input_dir = os.path.join(sydney_working_dir, "perspective_paragraph_feature")

    def work(self, job_id):
        features: List[ParagraphClaimPersFeature] = pickle.load(open(os.path.join(self.input_dir, str(job_id)), "rb"))

        writer = RecordWriterWrap(os.path.join(self.out_dir, str(job_id)))
        for f in features:
            encoded_list: List[OrderedDict] = format_feature(self.tokenizer, self.max_seq_length, f)
            foreach(writer.write_feature, encoded_list)
        writer.close()


if __name__ == "__main__":
    runner = JobRunner(sydney_working_dir, 605, "pc_paragraph_tfrecord", Worker)
    runner.start()