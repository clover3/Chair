import collections
import json
import os
from typing import List, Iterable
from typing import NamedTuple

from cache import load_pickle_from
from estimator_helper.output_reader import join_prediction_with_info
from misc_lib import group_by, find_max_idx, tprint, exist_or_mkdir, DataIDManager
from scipy_aux import logit_to_score_softmax
from tf_util.record_writer_wrap import write_records_w_encode_fn
from tlm.data_gen.classification_common import InstAsInputIds, encode_inst_as_input_ids
from tlm.data_gen.robust_gen.data_info_compression import decompress_seq
from tlm.robust.load import get_robust_splits


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


def enum_best_segments(pred_path, info) -> Iterable[SelectedSegment]:
    entries = join_prediction_with_info(pred_path, info)
    grouped = group_by(entries, lambda e: (e['query_id'], e['doc_id']))

    for key in grouped:
        sub_entries = grouped[key]

        def get_score(e):
            return logit_to_score_softmax(e['logits'])

        max_idx = find_max_idx(sub_entries, get_score)

        selected_raw_entry = sub_entries[max_idx]
        selected = decompress_entry(selected_raw_entry)

        assert len(selected['input_ids']) == len(selected['seg_ids'])
        ss = SelectedSegment(selected['input_ids'], selected['seg_ids'],
                             selected['passage_idx'], selected['label'],
                             selected['doc_id'], selected['query_id'],
                             get_score(selected))
        yield ss


def load_info_from_compressed(pickle_path):
    tprint("loading info pickle")
    output_d = {}
    data = load_pickle_from(pickle_path)
    tprint("loading info pickle")
    for data_id, value_d in data.items():
        new_entry = decompress_entry(value_d)
        output_d[data_id] = new_entry
    return output_d


def decompress_entry(value_d):
    new_entry = {}
    for key, value in value_d.items():
        if key == "seg_ids":
            new_entry[key] = decompress_seq(value)
        # elif key == "input_ids":
        #     new_entry['tokens'] = tokenizer.convert_ids_to_tokens(value)
        else:
            new_entry[key] = value
    return new_entry


def save_info(out_path, data_id_manager, job_id):
    info_dir = out_path + "_info"
    exist_or_mkdir(info_dir)
    info_path = os.path.join(info_dir, str(job_id) + ".info")
    json.dump(data_id_manager.id_to_info, open(info_path, "w"))


def generate_selected_training_data(split_no, score_dir, info_dir, max_seq_length, save_dir):
    train_items, held_out = get_robust_splits(split_no)
    print(train_items)
    exist_or_mkdir(save_dir)
    for key in train_items:
        data_id_manager = DataIDManager()
        info_path = os.path.join(info_dir, str(key))
        # info = load_combine_info_jsons(info_path, False, False)
        tprint("loading info: " + info_path)
        info = load_pickle_from(info_path)
        out_path = os.path.join(save_dir, str(key))
        pred_path = os.path.join(score_dir, str(key))
        tprint("data gen")
        itr = enum_best_segments(pred_path, info)

        insts = []
        for selected_segment in itr:
            data_id = data_id_manager.assign(selected_segment.to_info_d())
            ci = InstAsInputIds(
                selected_segment.input_ids,
                selected_segment.seg_ids,
                selected_segment.label,
                data_id)
            insts.append(ci)
            assert len(selected_segment.input_ids) <= max_seq_length

        def encode_fn(inst: InstAsInputIds) -> collections.OrderedDict:
            return encode_inst_as_input_ids(max_seq_length, inst)

        tprint("writing")
        write_records_w_encode_fn(out_path, encode_fn, insts, len(insts))
        save_info(save_dir, data_id_manager, str(key) + ".info")
