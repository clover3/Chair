import os

import numpy as np

from cache import load_pickle_from
from cpath import output_path


# if last_logits_d is not None:
#     logits1 = last_logits_d[name]
#     logits2 = logits_d[name]
#
#     input_ids1 = last_input_id_d[data_idx]
#     input_ids2 = input_id_d[data_idx]
#     if np.any(np.abs(logits1 - logits2) > 0.01):
#         print("logits differ")
#         print(name)
#         print(logits1)
#         print(logits2)
#     if not np.all(input_ids1 - input_ids2 == 0):

def input_id_differ(input_ids1, input_ids2):
    return not np.all(input_ids1 - input_ids2 == 0)


def logits_differ(logits1, logits2):
    return np.any(np.abs(logits1 - logits2) > 0.01)


def main():

    run_names = [
                "biobert_alamri1_deletion",
                 "biobert_alamri1_deletion_1",
                 "biobert_alamri1_deletion_2",
                 "biobert_alamri1_deletion_8"
    ]

    batch_size_list = [8, 1, 2, 8]
    num_deletion = 20
    d_per_run = {}
    for r_idx, name in enumerate(run_names):
        file_path = os.path.join(output_path, name, "0")
        batch_size = batch_size_list[r_idx]
        obj = load_pickle_from(file_path)
        print(name)
        data_id_d, input_id_d, logits_d = reorder_results(batch_size, num_deletion, obj)
        d_per_run[name] = data_id_d, input_id_d, logits_d

    run1 = run_names[2]
    run2 = run_names[3]
    data_id_d1, input_id_d1, logits_d1 = d_per_run[run1]
    data_id_d2, input_id_d2, logits_d2 = d_per_run[run2]
    for data_idx in range(0, 300):
        if input_id_differ(input_id_d1[data_idx], input_id_d2[data_idx]):
            print("data_id differ", data_id_d1[data_idx], data_id_d2[data_idx])

        for del_idx in range(21):
            logits1 = logits_d1[data_idx, del_idx]
            logits2 = logits_d2[data_idx, del_idx]
            if logits_differ(logits1, logits2):
                print("logits differ, del_idx={} logits1={} logits2={}".format(del_idx, logits1, logits2))




def reorder_results(batch_size, num_deletion, obj):
    logits_d = {}
    data_id_d = {}
    input_id_d = {}
    for b_idx, batch in enumerate(obj):
        if not len(batch["input_ids"]) == batch_size:
            print(len(batch["input_ids"]), batch_size)
            raise Exception
        assert len(batch["logits"]) == batch_size * (num_deletion + 1)
        batch_logits = batch["logits"]
        for in_batch_idx in range(batch_size):
            data_idx = b_idx * batch_size + in_batch_idx
            input_id = batch["input_ids"][in_batch_idx]
            data_id = batch["data_id"][in_batch_idx]
            input_id_d[data_idx] = input_id
            data_id_d[data_idx] = data_id
            for del_idx in range(num_deletion + 1):
                l_idx = in_batch_idx * (num_deletion + 1) + del_idx
                logits = batch_logits[l_idx]
                name = data_idx, del_idx
                logits_d[name] = logits
    return data_id_d, input_id_d, logits_d


if __name__ == "__main__":
    main()