import os
import pickle
from collections import OrderedDict
from typing import List, Dict

from arg.perspectives.pc_rel.pc_rel_structure import to_retrieval_format
from arg.perspectives.select_paragraph_perspective import ParagraphClaimPersFeature
from data_generator.common import get_tokenizer
from data_generator.job_runner import sydney_working_dir
from list_lib import foreach
from misc_lib import exist_or_mkdir, DataIDGen
from tf_util.record_writer_wrap import RecordWriterWrap


class PCRelTFRecordWorker:
    def __init__(self, input_job_name, out_dir):
        self.out_dir = out_dir
        self.info_out_dir = out_dir + "_info"
        exist_or_mkdir(out_dir)
        exist_or_mkdir(self.info_out_dir)

        self.tokenizer = get_tokenizer()
        self.max_seq_length = 512
        self.input_dir = os.path.join(sydney_working_dir, input_job_name)

    def work(self, job_id):
        features: List[ParagraphClaimPersFeature] = pickle.load(open(os.path.join(self.input_dir, str(job_id)), "rb"))

        info_d_all = {}
        data_id_base = job_id * 100000
        data_id_gen = DataIDGen(data_id_base)
        writer = RecordWriterWrap(os.path.join(self.out_dir, str(job_id)))
        for f in features:
            pair = to_retrieval_format(self.tokenizer, self.max_seq_length, data_id_gen, f)
            info_d: Dict = pair[0]
            f2: List[OrderedDict] = pair[1]

            info_d_all.update(info_d)
            foreach(writer.write_feature, f2)
        writer.close()

        pickle.dump(info_d_all, open(os.path.join(self.info_out_dir, str(job_id)), "wb"))
