import os
from typing import Tuple, Dict

from arg.perspectives.load import get_claim_perspective_label_dict
from arg.perspectives.pc_rel.rel_filtered import rel_filter
from arg.perspectives.types import CPIDPair, Logits, DataID
from cache import load_from_pickle
from data_generator.job_runner import sydney_working_dir
from data_generator.tokenizer_wo_tf import get_tokenizer
from tf_util.enum_features import load_record
from tf_util.record_writer_wrap import RecordWriterWrap


class PCRelFilterWorker:
    def __init__(self, input_job_name, pc_rel_info_pickle_name, out_dir):
        self.out_dir = out_dir
        self.tokenizer = get_tokenizer()
        self.max_seq_length = 512
        self.input_dir = os.path.join(sydney_working_dir, input_job_name)
        self.relevance_scores: Dict[DataID, Tuple[CPIDPair, Logits, Logits]] = load_from_pickle(pc_rel_info_pickle_name)
        self.cpid_to_label: Dict[CPIDPair, int] = get_claim_perspective_label_dict()

    def work(self, job_id):
        tfrecord_path = os.path.join(self.input_dir, str(job_id))
        features = load_record(tfrecord_path)

        save_path = os.path.join(self.out_dir, str(job_id))
        writer = RecordWriterWrap(save_path)
        for f in rel_filter(features, self.relevance_scores, self.cpid_to_label):
            writer.write_feature(f)
        writer.close()

