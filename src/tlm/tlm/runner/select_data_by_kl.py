import collections
import os
import random

import cache
from cie.arg import kl
from cpath import output_path
from data_generator.job_runner import JobRunner
from misc_lib import get_dir_files, TimeEstimator
from tf_util.enum_features import load_record
from tf_util.record_writer_wrap import RecordWriterWrap
from tlm.generate_bert_dummy import read_bert_data
from tlm.tfrecord_viewer import repack_features
from tlm.unigram_lm_from_tfrecord import LM, get_lm_tf


def gen_nli_lm():
    nli_train_fiction = os.path.join(output_path, "nli_tfrecord_cls_300", "train")
    lm_train_fiction = get_lm_tf(nli_train_fiction, 10000, True)
    cache.save_to_pickle(lm_train_fiction, "nli_lm")


def load_lm():
    f: collections.Counter = cache.load_from_pickle("nli_lm")
    return f

tf_record_dir = "/mnt/nfs/work3/youngwookim/data/bert_tf/unmasked_pair_x3"


def sample_median():
    # we don't want to make one of (bad/good) split to have shorter text than the other.
    all_scores = []
    scorer = get_lm_scorer()

    files = get_dir_files(tf_record_dir)
    random.shuffle(files)

    for file_path in files[:10]:
        tfrecord_itr = load_record(file_path)
        ticker = TimeEstimator(1000)
        for idx, inst in enumerate(tfrecord_itr):
            all_scores.append(scorer(inst))
            if idx > 1000:
                break
            ticker.tick()
    all_scores.sort()
    l = len(all_scores)
    print(l)
    mid = int(l/2)
    print(all_scores[mid])


def get_lm_scorer():
    target_lm_tf = load_lm()

    def score(inst):
        lm = LM(True)
        lm.update(inst["input_ids"].int64_list.value)
        div = kl.kl_divergence(target_lm_tf, lm.tf)
        return -div
    return score

class SplitWorker:
    def __init__(self, out_path_not_used):
        pass

    def work(self, job_id):
        split(job_id)

def split(job_id):
    threshold = -3.6859627006495708
    good_dir = "/mnt/nfs/work3/youngwookim/data/bert_tf/unmasked_good2"
    bad_dir = "/mnt/nfs/work3/youngwookim/data/bert_tf/unmasked_bad2"

    writer_good = RecordWriterWrap(os.path.join(good_dir, str(job_id)))
    writer_bad = RecordWriterWrap(os.path.join(bad_dir, str(job_id)))
    fn = os.path.join(tf_record_dir , str(job_id))
    tfrecord_itr = read_bert_data(fn)

    scorer = get_lm_scorer()

    for idx, inst in enumerate(tfrecord_itr):
        inst = repack_features(inst)
        if scorer(inst) > threshold:
            writer_good.write_feature(inst)
        else:
            writer_bad.write_feature(inst)

    writer_good.close()
    writer_bad.close()


if __name__ == "__main__":
    working_dir = "/mnt/nfs/work3/youngwookim/data/bert_tf/"
    runner = JobRunner(working_dir, 4000, "unmasked_split2", SplitWorker)
    runner.start()
