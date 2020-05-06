import os
from typing import Tuple, Dict

from arg.perspectives.load import get_claim_perspective_label_dict
from arg.perspectives.pc_rel.all_passage import collect_passages
from arg.perspectives.types import CPIDPair, Logits, DataID
from cache import load_from_pickle, load_pickle_from
from data_generator.common import get_tokenizer
from data_generator.job_runner import sydney_working_dir
from tf_util.enum_features import load_record
from tf_util.record_writer_wrap import RecordWriterWrap


class PCConcatRepWorker:
    def __init__(self, input_job_name, pc_rel_info_pickle_name, num_max_para, out_dir):
        self.out_dir = out_dir
        self.tokenizer = get_tokenizer()
        self.input_dir = os.path.join(sydney_working_dir, input_job_name)
        self.relevance_scores: Dict[DataID, Tuple[CPIDPair, Logits, Logits]] = load_from_pickle(pc_rel_info_pickle_name)
        self.cpid_to_label: Dict[CPIDPair, int] = get_claim_perspective_label_dict()
        self.num_max_para = num_max_para
        self.window_size = 354

    def work(self, job_id):
        tfrecord_path = os.path.join(self.input_dir, str(job_id))
        features = load_record(tfrecord_path)
        save_path = os.path.join(self.out_dir, str(job_id))
        writer = RecordWriterWrap(save_path)
        for f in collect_passages(features,
                                  self.relevance_scores,
                                  self.cpid_to_label,
                                  self.num_max_para,
                                  self.window_size):
            writer.write_feature(f)
        writer.close()


class PCConcatFocusWorker(PCConcatRepWorker):
    def __init__(self, rel_ex_score_dir, input_job_name, pc_rel_info_pickle_name, num_max_para, out_dir):
        super(PCConcatFocusWorker, self).__init__(input_job_name, pc_rel_info_pickle_name, num_max_para, out_dir)
        self.rel_ex_score_dir = rel_ex_score_dir

    def work(self, job_id):
        tfrecord_path = os.path.join(self.input_dir, str(job_id))
        features = load_record(tfrecord_path)
        save_path = os.path.join(self.out_dir, str(job_id))
        rel_score_path = os.path.join(self.rel_ex_score_dir, str(job_id))
        rel_score = load_pickle_from(rel_score_path)
        writer = RecordWriterWrap(save_path)
        for f in collect_passages(features,
                                  self.relevance_scores,
                                  self.cpid_to_label,
                                  self.num_max_para,
                                  self.window_size,
                                  rel_score):
            writer.write_feature(f)
        writer.close()