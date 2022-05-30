import os

from dataset_specific.mnli.mnli_reader import MNLIReader
from epath import job_man_dir
from job_manager.job_runner3 import PartitionSpec, PartitionDataSpec


def get_mnli_partition_spec(split):
    reader = MNLIReader()
    data_size = reader.get_data_size(split)
    num_record_per_job = 1000
    ps = PartitionSpec.from_total_size(data_size, num_record_per_job)
    return ps


def get_mnli_spacy_ps(split) -> PartitionDataSpec:
    ps = get_mnli_partition_spec(split)
    job_name = "mnli_spacy_tokenize_{}".format(split)
    return partition_data_spec_job_man_dir(ps, job_name)


def partition_data_spec_job_man_dir(ps, job_name) -> PartitionDataSpec:
    dir_path = os.path.join(job_man_dir, job_name)
    return PartitionDataSpec.build(ps, dir_path)


