import os

from data_generator.bert_input_splitter import split_p_h_with_input_ids, SEPNotFound, SEP_ID
from data_generator.job_runner import WorkerInterface
from epath import job_man_dir
from tf_util.tfrecord_convertor import extract_convertor_w_float


def show_as_string(int_list):
    return " ".join(map(str, int_list))


def string_to_int_list(s):
    return list(map(int, s.split()))


def drop_entity_from_query(qe_input_ids, qe_segment_ids):
    try:
        entity, query = split_p_h_with_input_ids(qe_input_ids, qe_input_ids)
    except SEPNotFound:
        entity = []
        query = []

    entity_as_string = show_as_string(entity)
    query_as_string = show_as_string(query)
    maybe_mask_id = 103
    entity_less_query = query_as_string.replace(entity_as_string, str(maybe_mask_id))

    CLS_ID = qe_input_ids[0]
    new_seg1 = string_to_int_list(entity_less_query)
    pad_len = len(qe_input_ids) - len(new_seg1) - 2
    new_seg1_input_ids = [CLS_ID] + new_seg1 + [SEP_ID] + [0] * pad_len
    new_seg1_segment_ids = [0] * len(qe_segment_ids)
    return new_seg1_input_ids, new_seg1_segment_ids


def convert(d_pair):
    d_int, d_float = d_pair
    out_d_int = {}
    out_d_int["d_e_input_ids"] = d_int["d_e_input_ids"]
    out_d_int["d_e_segment_ids"] = d_int["d_e_input_ids"]
    out_d_int["data_id"] = d_int["data_id"]

    qe_input_ids = d_int["q_e_input_ids"]
    qe_segment_ids = d_int["q_e_segment_ids"]
    new_seg1_input_ids, new_seg1_segment_ids = drop_entity_from_query(qe_input_ids, qe_segment_ids)
    out_d_int['q_e_input_ids'] = new_seg1_input_ids
    out_d_int['q_e_segment_ids'] = new_seg1_segment_ids

    out_d_float = {}
    out_d_float["label_ids"] = d_float["label_ids"]
    return out_d_int, out_d_float


class Worker(WorkerInterface):
    def __init__(self, source_job, out_dir):
        self.out_dir = out_dir
        self.source_job = source_job

    def work(self, job_id):
        input_tfrecord_path = os.path.join(job_man_dir, self.source_job, str(job_id))
        save_tfrecord_path = os.path.join(self.out_dir, str(job_id))
        if os.path.exists(input_tfrecord_path):
            extract_convertor_w_float(input_tfrecord_path, save_tfrecord_path, convert)
        else:
            print("File does not exists at ", input_tfrecord_path)