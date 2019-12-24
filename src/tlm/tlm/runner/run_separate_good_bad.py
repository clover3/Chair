import os
import pickle
import random

import numpy as np

from data_generator.job_runner import JobRunner
from misc_lib import get_dir_files
from tf_util.record_writer_wrap import RecordWriterWrap
from tlm.estimator_prediction_viewer import flatten_batches
from tlm.tfrecord_viewer import read_bert_data, repack_features

working_dir = "/mnt/nfs/work3/youngwookim/data/bert_tf/"


def get_prediction_dir(working_dir):
    return os.path.join(working_dir, "loss_predictor_predictions")



def score_0(prob1, prob2):
    p1 = np.sum(prob1, axis=1)
    p2 = np.sum(prob2, axis=1)
    return -(p1-p2)


def score_1(prob1, prob2):
    divider = np.maximum(prob1, prob2)
    eps = 0.0001
    divider += eps
    raw_scores = np.divide(prob2, divider)
    scores = np.sum(raw_scores, axis=1)
    return scores


threshold_score_0 = -92.72
threshold_score_1 = 259.27118
score_option = 1

if score_option == 0:
    scorer = score_0
    threshold = threshold_score_0
elif score_option == 1:
    scorer = score_1
    threshold = threshold_score_1
else:
    raise Exception("Not def")



def sample_median():
    # we don't want to make one of (bad/good) split to have shorter text than the other.
    files = get_dir_files(get_prediction_dir(working_dir))
    random.shuffle(files)

    all_scores = []
    for file_path in files[:10]:
        data = pickle.load(open(file_path, "rb"))
        data = flatten_batches(data)
        t = scorer(data["prob1"], data["prob2"])
        all_scores.extend(t)

    all_scores.sort()
    l = len(all_scores)
    print(l)
    mid = int(l/2)
    print(all_scores[mid])


tf_record_dir = "/mnt/nfs/work3/youngwookim/data/bert_tf/unmasked_pair_x3"

class SplitWorker:
    def __init__(self, out_path_not_used):
        pass

    def work(self, job_id):
        split(job_id)


def split(job_id):

    file_path = os.path.join(get_prediction_dir(working_dir), str(job_id))
    if not os.path.exists(file_path):
        return
    data = pickle.load(open(file_path, "rb"))
    data = flatten_batches(data)
    scores = scorer(data["prob1"], data["prob2"])
    good_list = []
    bad_list = []
    for i, score in enumerate(scores):
        if score > threshold:
            good_list.append(i)
        else:
            bad_list.append(i)

    good_dir = "/mnt/nfs/work3/youngwookim/data/bert_tf/unmasked_good1"
    bad_dir = "/mnt/nfs/work3/youngwookim/data/bert_tf/unmasked_bad1"

    writer_good = RecordWriterWrap(os.path.join(good_dir, str(job_id)))
    writer_bad  = RecordWriterWrap(os.path.join(bad_dir, str(job_id)))
    fn = os.path.join(tf_record_dir , str(job_id))
    tfrecord_itr = read_bert_data(fn)
    num_scores = len(scores)
    for idx, inst in enumerate(tfrecord_itr):
        inst = repack_features(inst)
        if idx in good_list:
            writer_good.write_feature(inst)
        elif idx in bad_list:
            writer_bad.write_feature(inst)
        else:
            if idx < num_scores:
                raise Exception("Data {} not found".format(idx))

    writer_good.close()
    writer_bad.close()


if __name__ == "__main__":
    #sample_median()
    runner = JobRunner(working_dir, 4000, "unmasked_split1", SplitWorker)
    runner.start()

