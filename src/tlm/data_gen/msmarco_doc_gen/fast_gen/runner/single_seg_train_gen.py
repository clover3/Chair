import argparse
import os
import sys

from data_generator.job_runner import JobRunner
from dataset_specific.msmarco.common import train_query_group_len
from epath import job_man_dir
from misc_lib import exist_or_mkdir
from tlm.data_gen.msmarco_doc_gen.fast_gen.best_seg_train_gen import SingleSegTrainGen
from tlm.data_gen.msmarco_doc_gen.fast_gen.collect_best_seg_prediction import BestSegCollector


def main():
    job_name = "MMD_train_single_seg"
    out_dir = os.path.join(job_man_dir, job_name)
    exist_or_mkdir(out_dir)
    worker = SingleSegTrainGen(512, out_dir)
    for job_id in range(178, train_query_group_len):
        print("job_id:", job_id)
        worker.work(job_id)


if __name__ == "__main__":
    main()
