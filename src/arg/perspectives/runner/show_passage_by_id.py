import os
from typing import Tuple, Dict

import numpy as np

from arg.perspectives.types import DataID, Logits, CPIDPair
from cache import load_from_pickle
from cpath import pjoin, output_path, data_path
from data_generator.tokenizer_wo_tf import pretty_tokens, get_tokenizer
from tf_util.enum_features import load_record
from tlm.data_gen.feature_to_text import take


def split_with_segment_ids(input_ids, segment_ids):
    state = 0
    for i in range(len(segment_ids)):
        if state == 0 and segment_ids[i] == 1:
            p1 = i
            state = 1
        elif state == 1 and segment_ids[i] == 0:
            p2 = i
    return input_ids[:p1], input_ids[p1:p2]

def do_print(cpid: CPIDPair,
             doc_index: Dict[CPIDPair, Tuple[DataID, Logits, Logits]],
             tokenizer
          ):

    data_id_to_input_ids = None
    loaded_file_no = None

    def load(file_no):
        path = os.path.join(data_path, "pc_rel_tfrecord_dev", str(file_no))
        d = {}
        for feature in load_record(path):
            data_id = take(feature["data_id"])[0]
            input_ids = take(feature["input_ids"])
            segment_ids = take(feature["segment_ids"])
            d[data_id] = input_ids, segment_ids
            print(data_id)
        print("loaded {} data".format(len(d)))
        return d

    def print_ids(ids):
        print(pretty_tokens(tokenizer.convert_ids_to_tokens(ids), True))

    for entry in doc_index[cpid]:
        data_id, c_logits, p_logits = entry
        file_no = int(data_id / 100000)
        if loaded_file_no != file_no:
            data_id_to_input_ids = load(file_no)
            loaded_file_no = file_no

        try:
            print(data_id, "aa")
            input_ids, segment_ids = data_id_to_input_ids[data_id]
            seg1, seg2 = split_with_segment_ids(input_ids, segment_ids)

            p_pred = np.argmax(p_logits)
            c_pred = np.argmax(c_logits)
            print("c_pred/p_pred", c_pred, p_pred)
            print_ids(seg1)
            print_ids(seg2)

        except KeyError as e:
            print("KyError", e)





def main():
    info = load_from_pickle("pc_rel_dev_info_all")
    prediction_path = pjoin(output_path, "pc_rel_dev")
    rel_info: Dict[DataID, Tuple[CPIDPair, Logits, Logits]] = load_from_pickle("pc_rel_dev_with_cpid")
    #rel_info: Dict[DataID, Tuple[CPIDPair, Logits, Logits]] = combine_pc_rel_with_cpid(prediction_path, info)

    doc_index = reverse_index(rel_info)
    tokenizer = get_tokenizer()

    while True:
        s = input()
        os.system('cls')
        cid, pid = s.split()
        cid = int(cid)
        pid = int(pid)
        cpid = CPIDPair((cid, pid))
        do_print(cpid, doc_index, tokenizer)


def reverse_index(rel_info):
    doc_index = {}
    for data_id, tuple in rel_info.items():
        tuple: Tuple[CPIDPair, Logits, Logits] = tuple
        cpid, c_logits, p_logits = tuple

        if cpid not in doc_index:
            doc_index[cpid] = []

        doc_index[cpid].append((data_id, c_logits, p_logits))
    return doc_index


if __name__ == "__main__":
    main()