import collections
import json
import os
from typing import List, Iterable, Dict
from typing import NamedTuple

from cache import load_pickle_from
from list_lib import left
from misc_lib import group_by, find_max_idx, tprint, exist_or_mkdir, DataIDManager
from scipy_aux import logit_to_score_softmax
from tf_util.record_writer_wrap import write_records_w_encode_fn
from tlm.data_gen.base import pad0
from tlm.data_gen.classification_common import InstAsInputIds, encode_inst_as_input_ids
from tlm.data_gen.robust_gen.data_info_compression import decompress_seq
from tlm.estimator_output_reader import join_prediction_with_info
from tlm.robust.load import get_robust_splits, robust_query_intervals


class SelectedSegment(NamedTuple):
    input_ids: List[int]
    seg_ids: List[int]
    passage_idx: int
    label: int
    doc_id: str
    query_id: str
    prob: float

    def to_info_d(self):
        return {
            'query_id': self.query_id,
            'doc_id': self.doc_id,
            'passage_idx': self.passage_idx,
            'label': self.label,
        }


def enum_best_segments(pred_path, info) -> Iterable[Dict]:
    entries = join_prediction_with_info(pred_path, info)
    grouped = group_by(entries, lambda e: (e['query_id'], e['doc_id']))

    for key in grouped:
        sub_entries = grouped[key]

        def get_score(e):
            return logit_to_score_softmax(e['logits'])

        max_idx = find_max_idx(get_score, sub_entries)

        selected_raw_entry = sub_entries[max_idx]
        yield selected_raw_entry


def load_info_from_compressed(pickle_path):
    tprint("loading info pickle")
    output_d = {}
    data = load_pickle_from(pickle_path)
    tprint("decompressing...")
    for data_id, value_d in data.items():
        new_entry = decompress_seg_ids_entry(value_d)
        output_d[data_id] = new_entry
    return output_d


def decompress_seg_ids_entry(value_d):
    if 'decompressed' in value_d:
        return value_d
    raw_seg_id = value_d["seg_ids"]
    value_d["seg_ids"] = decompress_seq(raw_seg_id)
    value_d['decompressed'] = True
    return value_d


def save_info(out_path, data_id_manager, job_id):
    info_dir = out_path + "_info"
    exist_or_mkdir(info_dir)
    info_path = os.path.join(info_dir, str(job_id) + ".info")
    json.dump(data_id_manager.id_to_info, open(info_path, "w"))


def generate_selected_training_data_loop(split_no,
                                         score_dir,
                                         info_dir,
                                         max_seq_length,
                                         save_dir,
                                         generate_selected_training_data_fn
                                         ):
    train_items, held_out = get_robust_splits(split_no)
    print(train_items)
    exist_or_mkdir(save_dir)
    for key in train_items:
        info_path = os.path.join(info_dir, str(key))
        # info = load_combine_info_jsons(info_path, False, False)
        tprint("loading info: " + info_path)
        info = load_pickle_from(info_path)
        # info = load_info_from_compressed(info_path)
        generate_selected_training_data_fn(info, key, max_seq_length, save_dir, score_dir)

def generate_selected_training_data_for_many_runs(target_data_idx,
                                                  info_dir,
                                                  max_seq_length,
                                                  score_and_save_dir: List,
                                                  generate_selected_training_data_fn
                                                  ):
    interval_start_list = left(robust_query_intervals)
    key = interval_start_list[target_data_idx]
    info_path = os.path.join(info_dir, str(key))
    tprint("loading info: " + info_path)
    info = load_pickle_from(info_path)
    for score_dir, save_dir in score_and_save_dir:
        exist_or_mkdir(save_dir)
        tprint(save_dir)
        generate_selected_training_data_fn(info, key, max_seq_length, save_dir, score_dir)


def generate_selected_training_data(info, key, max_seq_length, save_dir, score_dir):
    data_id_manager = DataIDManager(0, 1000000)
    out_path = os.path.join(save_dir, str(key))
    pred_path = os.path.join(score_dir, str(key))
    tprint("data gen")
    itr = enum_best_segments(pred_path, info)
    insts = []
    for selected_entry in itr:
        selected = decompress_seg_ids_entry(selected_entry)
        assert len(selected['input_ids']) == len(selected['seg_ids'])

        selected['input_ids'] = pad0(selected['input_ids'], max_seq_length)
        selected['seg_ids'] = pad0(selected['seg_ids'], max_seq_length)
        # data_id = data_id_manager.assign(selected_segment.to_info_d())
        data_id = 0
        ci = InstAsInputIds(
            selected['input_ids'],
            selected['seg_ids'],
            selected['label'],
            data_id)
        insts.append(ci)

    def encode_fn(inst: InstAsInputIds) -> collections.OrderedDict:
        return encode_inst_as_input_ids(max_seq_length, inst)

    tprint("writing")
    write_records_w_encode_fn(out_path, encode_fn, insts, len(insts))
    save_info(save_dir, data_id_manager, str(key) + ".info")
