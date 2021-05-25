

import argparse
import os
import sys

from data_generator.job_runner import JobRunner
from dataset_specific.msmarco.common import train_query_group_len
from epath import job_man_dir
from misc_lib import exist_or_mkdir
from tlm.data_gen.msmarco_doc_gen.fast_gen.best_seg_prediction_gen import BestSegmentPredictionGen
from tlm.data_gen.msmarco_doc_gen.fast_gen.best_seg_train_gen import BestSegTrainGen
from tlm.data_gen.msmarco_doc_gen.fast_gen.collect_best_seg_prediction import BestSegCollector
from tlm.data_gen.msmarco_doc_gen.fast_gen.seg_resource import SegmentResourceLoader

parser = argparse.ArgumentParser(description='')
parser.add_argument("--prediction_dir")
parser.add_argument("--score_type")
parser.add_argument("--job_id")
parser.add_argument("--model_name")


def main1():
    args = parser.parse_args(sys.argv[1:])
    info_dir = os.path.join(job_man_dir, "best_seg_prediction_train_info")
    bsc = BestSegCollector(
        info_dir,
        args.prediction_dir,
        args.score_type)

    d = bsc.get_best_seg_info(0)


def main3():
    srl = SegmentResourceLoader(job_man_dir, "train")
    sr_per_query = srl.load_for_qid("1000633")


def main():
    split = "train"
    bpg = BestSegmentPredictionGen(512, split, "/tmp/st")
    bpg.work(0)


if __name__ == "__main__":
    main()