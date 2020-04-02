import os
import pickle
from collections import OrderedDict
from typing import List

from arg.perspectives.encode_paragraph_feature_to_tfrecord import format_paragraph_features
from arg.perspectives.select_paragraph_perspective import ParagraphClaimPersFeature
from data_generator.common import get_tokenizer
from data_generator.job_runner import sydney_working_dir
from list_lib import foreach
from misc_lib import exist_or_mkdir
from tf_util.record_writer_wrap import RecordWriterWrap


class ParagraphTFRecordWorker:
    def __init__(self, input_job_name, out_dir):
        self.out_dir = out_dir
        exist_or_mkdir(out_dir)
        self.tokenizer = get_tokenizer()
        self.max_seq_length = 512
        self.input_dir = os.path.join(sydney_working_dir, input_job_name)

    def work(self, job_id):
        features: List[ParagraphClaimPersFeature] = pickle.load(open(os.path.join(self.input_dir, str(job_id)), "rb"))

        writer = RecordWriterWrap(os.path.join(self.out_dir, str(job_id)))
        for f in features:
            encoded_list: List[OrderedDict] = format_paragraph_features(self.tokenizer, self.max_seq_length, f)
            foreach(writer.write_feature, encoded_list)
        writer.close()