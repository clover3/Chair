import csv

from dataset_specific.msmarco.passage.grouped_reader import get_train_neg5_sample_path
from dataset_specific.msmarco.passage.passage_resource_loader import MMPPosNegSampler
from misc_lib import TimeEstimator
from trainer_v2.per_project.transparency.mmp.data_enum import enum_pos_neg_pointwise_per_partition
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.split_iter import get_valid_mmp_partition


def main():
    pos_neg_sampler = MMPPosNegSampler()
    split = "train"
    n_neg = 5
    ticker = TimeEstimator(110)
    for partition_no in get_valid_mmp_partition(split):
        itr = enum_pos_neg_pointwise_per_partition(pos_neg_sampler, partition_no, n_neg)
        save_path = get_train_neg5_sample_path(partition_no)
        tsv_writer = csv.writer(open(save_path, "w", newline=""), delimiter="\t")
        for query, doc, label in itr:
            tsv_writer.writerow([query, doc, str(label)])
        ticker.tick()
        break


if __name__ == "__main__":
    main()
