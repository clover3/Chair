import argparse
import os
import sys

from data_generator.job_runner import JobRunner
from dataset_specific.msmarco.common import train_query_group_len
from epath import job_man_dir
from misc_lib import exist_or_mkdir
from tlm.data_gen.msmarco_doc_gen.fast_gen.best_seg_train_gen import BestSegTrainGen
from tlm.data_gen.msmarco_doc_gen.fast_gen.collect_best_seg_prediction import BestSegCollector


parser = argparse.ArgumentParser(description='')
parser.add_argument("--prediction_dir")
parser.add_argument("--score_type")
parser.add_argument("--job_id")
parser.add_argument("--model_name")


def main():
    args = parser.parse_args(sys.argv[1:])
    info_dir = os.path.join(job_man_dir, "MMD_best_seg_prediction_train_info")
    bsc = BestSegCollector(
        info_dir,
        args.prediction_dir,
        args.score_type)

    job_id = int(args.job_id)
    model_name = args.model_name
    job_name = "MMD_train_best_by_{}".format(model_name)
    out_dir = os.path.join(job_man_dir, job_name)
    exist_or_mkdir(out_dir)
    worker = BestSegTrainGen(512, bsc, out_dir)
    worker.work(job_id)


if __name__ == "__main__":
    main()