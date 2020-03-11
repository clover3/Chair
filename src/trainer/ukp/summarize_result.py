import os
import pickle

import numpy as np

from misc_lib import get_dir_files


## This code only runs on gypsum

def summarize_prediction(obj):
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


def summarize_logits(obj):
    result = []
    for idx, doc in enumerate(obj):
        if not doc:
            continue
        logit_list = []
        for sent_info in doc:
            logits = sent_info[0]
            logit_list.append(logits)

        result.append((idx, logit_list))
    return result


def summarize_runner(summarizer, out_root):
    dir_root = "/mnt/nfs/scratch1/youngwookim/data/clueweb12_10000_pred_ex"
    for file_path in get_dir_files(dir_root):
        try:
            if "abortion" not in file_path:
                continue
            print(file_path)
            file_name = os.path.basename(file_path)
            obj = pickle.load(open(file_path, "rb"))
            r = summarizer(obj)
            out_path = os.path.join(out_root, file_name)
            pickle.dump(r, open(out_path, "wb"))
        except Exception as e:
            print(e)
            pass


def job1():
    summarizer = summarize_prediction
    out_root = "/mnt/nfs/scratch1/youngwookim/data/clueweb12_10000_pred_ex_summary"
    summarize_runner(summarizer, out_root)


def job2():
    out_root = "/mnt/nfs/scratch1/youngwookim/data/clueweb12_10000_pred_ex_summary_w_logit"
    summarize_runner(summarize_logits, out_root)

if __name__ == "__main__":
    job2()