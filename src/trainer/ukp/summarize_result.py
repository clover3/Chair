import os
import pickle

import numpy as np

from misc_lib import get_dir_files


def summarize(obj):
    result = []

    for idx, doc in enumerate(obj):
        if not doc:
            continue
        pred = []
        for sent_info in doc:
            logits = sent_info[0]
            y = np.argmax(logits)
            pred.append(y)

        result.append((idx, pred))

    return result


def read_do_save():
    dir_root = "/mnt/nfs/scratch1/youngwookim/data/clueweb12_10000_pred_ex"
    out_root = "/mnt/nfs/scratch1/youngwookim/data/clueweb12_10000_pred_ex_summary"
    for file_path in get_dir_files(dir_root):
        print(file_path)
        try:
            if "abortion" in file_path:
                continue
            file_name = os.path.basename(file_path)
            obj = pickle.load(open(file_path, "rb"))
            r = summarize(obj)
            out_path = os.path.join(out_root, file_name)
            pickle.dump(r, open(out_path, "wb"))
        except:
            pass

if __name__ == "__main__":
    read_do_save()