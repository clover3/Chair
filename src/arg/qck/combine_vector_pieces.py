import json
import os
import sys
from collections import OrderedDict
from typing import List, Iterable, Dict, Tuple

import numpy as np

from arg.qck.decl import qck_convert_map
from arg.qck.prediction_reader import load_combine_info_jsons
from estimator_helper.output_reader import join_prediction_with_info
from list_lib import dict_value_map
from misc_lib import group_by, exist_or_mkdir, DataIDManager
from tf_util.record_writer_wrap import write_records_w_encode_fn
from tlm.data_gen.bert_data_gen import create_int_feature, create_float_feature

Vectors = List[np.array]
Label = int


def write_to_file(output_path, g2: Iterable[Tuple[int, Tuple[Vectors, Label]]], max_entries):

    def encode(e: Tuple[int, Tuple[Vectors, Label]]) -> OrderedDict:
        data_id, (vector, label) = e
        features = OrderedDict()
        features['label_ids'] = create_int_feature([label])
        features['data_id'] = create_int_feature([data_id])

        vector = np.stack(vector, axis=0) # [n_entries, seq-length, hidden_unit]
        vector = vector[:max_entries]
        vector_len, seq_len, hidden_unit = np.shape(vector)
        valid_mask = np.ones([vector_len, seq_len, 1], np.int)
        if len(vector) < max_entries:
            pad_len = max_entries - len(vector)
            vector = np.concatenate([vector, np.zeros([pad_len, seq_len, hidden_unit])], axis=0)
            valid_mask = np.concatenate([valid_mask, np.zeros([pad_len, seq_len, 1], np.int)], axis=0)

        v = np.reshape(vector, [-1]) # [n_entries * seq_length]
        valid_mask = np.reshape(valid_mask, [-1])  # [n_entries * seq_length]
        features['vectors'] = create_float_feature(v)
        features['valid_mask'] = create_int_feature(valid_mask)
        return features

    write_records_w_encode_fn(output_path, encode, g2)


def do_job(input_dir, output_dir, info_dir,
           label_info_path, max_entries,
           job_id):

    exist_or_mkdir(output_dir)
    info_output_dir = output_dir + "_info"
    exist_or_mkdir(info_output_dir)

    label_info: List[Tuple[str, str, int]] = json.load(open(label_info_path, "r"))
    label_info_d = {(str(a), str(b)) : c for a, b, c in label_info}

    pred_path = os.path.join(input_dir, str(job_id) + ".score")
    #info_path = os.path.join(info_dir, str(job_id) + ".info")
    info_path = info_dir
    output_path = os.path.join(output_dir, str(job_id))
    info_output_path = os.path.join(info_output_dir, str(job_id))
    info = load_combine_info_jsons(info_path, qck_convert_map, True)
    fetch_field_list = ["vector", "data_id"]

    predictions = join_prediction_with_info(pred_path, info, fetch_field_list)

    def get_qid(entry):
        return entry['query'].query_id

    def get_candidate_id(entry):
        return entry['candidate'].id

    def pair_id(entry) -> Tuple[str, str]:
        return get_qid(entry), get_candidate_id(entry)

    groups: Dict[Tuple[str, str], List[Dict]] = group_by(predictions, pair_id)

    def get_new_entry(entries: List[Dict]):
        if not entries:
            return None
        vectors: Vectors = list([e['vector'] for e in entries])
        key = pair_id(entries[0])
        if key in label_info_d:
            label: Label = label_info_d[key]
        else:
            label: Label = 0

        return vectors, label

    g2: Dict[Tuple[str, str], Tuple[Vectors, Label]] = dict_value_map(get_new_entry, groups)
    base = 100 * 1000 * job_id
    max_count = 100 * 1000 * (job_id + 1)
    data_id_manager = DataIDManager(base, max_count)

    def get_out_itr() -> Iterable[Tuple[int, Tuple[Vectors, Label]]]:
        for key, data in g2.items():
            qid, cid = key
            data_info = {
                'qid': qid,
                'cid': cid,
            }
            data_id = data_id_manager.assign(data_info)
            yield data_id, data

    write_to_file(output_path, get_out_itr(), max_entries)
    json.dump(data_id_manager.id_to_info, open(info_output_path, "w"))


def main():
    run_config = json.load(open(sys.argv[1], "r"))
    job_id = int(sys.argv[2])
    do_job(run_config['input_dir'],
           run_config['output_dir'],
           run_config['info_dir'],
           run_config['label_info'],
           run_config['max_entries'],
           job_id)


if __name__ == "__main__":
    main()
