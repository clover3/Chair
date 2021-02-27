import json
import pickle
import sys
from typing import List, Tuple

import numpy as np

from data_generator.tokenizer_wo_tf import get_tokenizer


def compress_seq(raw_seq) -> List[Tuple[int, int]]:
    prev_val = None
    compressed = []
    n_val = 0
    for i in range(len(raw_seq)):
        cur_val = raw_seq[i]
        if prev_val == None:
            prev_val = cur_val
            n_val = 1
        else:
            if prev_val != cur_val:
                compressed.append((prev_val, n_val))
                prev_val = cur_val
                n_val = 1
            else:
                n_val += 1

    compressed.append((cur_val, n_val))
    return compressed


def decompress_seq(compressed: List[Tuple[int, int]]) -> List[int]:
    output = []
    for val, n in compressed:
        for _ in range(n):
            output.append(val)
    return output

##
def compress_input_mask_segment_ids(input_path, output_path):
    j = json.load(open(input_path, "r", encoding="utf-8"))
    tokenizer = get_tokenizer()
    new_j = {}
    for data_id, value_d in j.items():
        new_entry = {}
        for key in value_d:
            if key == "seg_ids":
                seg_ids = value_d[key]
                new_seg_ids = compress_seq(seg_ids)
                new_entry[key] = new_seg_ids
                ##
                assert np.all(np.array(decompress_seq(new_seg_ids)) == np.array(seg_ids))
            elif key == "tokens":
                tokens = value_d[key]
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                new_entry['input_ids'] = input_ids
            else:
                new_entry [key] = value_d[key]
        new_j[data_id] = new_entry
    pickle.dump(new_j, open(output_path, "wb"))


def main():
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    compress_input_mask_segment_ids(input_path, output_path)




if __name__ == "__main__":
    main()