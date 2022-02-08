import collections
import os
from typing import Iterable, Dict

from estimator_helper.output_reader import join_prediction_with_info
from misc_lib import group_by, find_max_idx, tprint, DataIDManager
from scipy_aux import logit_to_score_softmax
from tf_util.record_writer_wrap import write_records_w_encode_fn
from tlm.data_gen.base import pad0
from tlm.data_gen.classification_common import InstAsInputIds, encode_inst_as_input_ids
from tlm.data_gen.robust_gen.select_supervision.read_score import decompress_seg_ids_entry, save_info


def enum_best_segments_only_pos(pred_path, info) -> Iterable[Dict]:
    entries = join_prediction_with_info(pred_path, info)
    grouped = group_by(entries, lambda e: (e['query_id'], e['doc_id']))

    for key in grouped:
        sub_entries = grouped[key]

        def get_score(e):
            return logit_to_score_softmax(e['logits'])

        label = sub_entries[0]['label']
        if label:
            max_idx = find_max_idx(get_score, sub_entries)
        else:
            max_idx = 0

        selected_raw_entry = sub_entries[max_idx]
        yield selected_raw_entry


def enum_best_segments_only_neg(pred_path, info) -> Iterable[Dict]:
    entries = join_prediction_with_info(pred_path, info)
    grouped = group_by(entries, lambda e: (e['query_id'], e['doc_id']))

    for key in grouped:
        sub_entries = grouped[key]

        def get_score(e):
            return logit_to_score_softmax(e['logits'])

        label = sub_entries[0]['label']
        if label:
            max_idx = 0
        else:
            max_idx = find_max_idx(get_score, sub_entries)

        selected_raw_entry = sub_entries[max_idx]
        yield selected_raw_entry


def enum_best_segments_always(pred_path, info) -> Iterable[Dict]:
    entries = join_prediction_with_info(pred_path, info)
    grouped = group_by(entries, lambda e: (e['query_id'], e['doc_id']))

    for key in grouped:
        sub_entries = grouped[key]

        def get_score(e):
            return logit_to_score_softmax(e['logits'])

        max_idx = find_max_idx(get_score, sub_entries)

        selected_raw_entry = sub_entries[max_idx]
        yield selected_raw_entry


def generate_selected_training_data_ablation(option):
    if option == "positive":
        enum_best_segments = enum_best_segments_only_pos
    elif option == "negative":
        enum_best_segments = enum_best_segments_only_neg
    elif option == "always":
        enum_best_segments = enum_best_segments_always
    else:
        assert False

    def generate_selected_training_data_ablation_only_pos(info, key, max_seq_length, save_dir, score_dir):
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
    return generate_selected_training_data_ablation_only_pos