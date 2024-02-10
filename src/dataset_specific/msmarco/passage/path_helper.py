import itertools
import json

from cpath import data_path, output_path
from misc_lib import path_join
from table_lib import tsv_iter
from typing import List, Iterable, Callable, Dict, Tuple, Set
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator

def get_mmp_train_grouped_sorted_path(job_no):
    quad_tsv_path = path_join(data_path, "msmarco", "passage", "group_sorted_10K", str(job_no))
    return quad_tsv_path


def get_mmp_grouped_sorted_path(split, job_no):
    if split == "train":
        return get_mmp_train_grouped_sorted_path(job_no)
    else:
        quad_tsv_path = path_join(data_path, "msmarco", "passage", f"{split}_group_sorted_10K", str(job_no))
    return quad_tsv_path


TREC_DL_2019 = "TREC_DL_2019"
TREC_DL_2020 = "TREC_DL_2020"


def get_mmp_test_queries_path(dataset_name):
    if dataset_name == TREC_DL_2019:
        quad_tsv_path = path_join(data_path, "msmarco", "passage", dataset_name, "queries_2019", "raw.tsv")
    elif dataset_name == TREC_DL_2020:
        quad_tsv_path = path_join(data_path, "msmarco", "passage", dataset_name, "queries_2020", "raw.tsv")
    else:
        raise ValueError()
    return quad_tsv_path


def load_mmp_test_queries(dataset_name) -> List[Tuple[str, str]]:
    itr = tsv_iter(get_mmp_test_queries_path(dataset_name))
    return list(itr)


def load_mmp_queries(dataset_name) -> List[Tuple[str, str]]:
    if dataset_name.startswith("dev"):
        tsv_path = path_join(data_path, "msmarco", dataset_name, "queries.tsv")
    else:
        tsv_path = get_mmp_test_queries_path(dataset_name)

    itr = tsv_iter(tsv_path)
    return list(itr)


def get_mmp_test_qrel_binary_json_path(dataset_name):
    return path_join(data_path, "msmarco", "passage", dataset_name, "qrel_binary.json")


def get_mmp_test_qrel_json_path(dataset_name):
    return path_join(data_path, "msmarco", "passage", dataset_name, "qrel.json")


def load_mmp_test_qrel_binary_json(dataset_name):
    return json.load(open(get_mmp_test_qrel_binary_json_path(dataset_name), "r"))


def load_mmp_test_qrel_json(dataset_name):
    return json.load(open(get_mmp_test_qrel_json_path(dataset_name), "r"))


# 397,756,691

MSMARCO_PASSAGE_TRIPLET_SIZE = 397756691

def get_train_triples_path():
    tsv_path = path_join(data_path, "msmarco", "triples.train.full.tsv.gz")
    return tsv_path


def get_train_triples_partition_path(part_no):
    return path_join(data_path, "msmarco", "triples.train.full", str(part_no))


def iter_train_triples_partition(part_no):
    tsv_path = get_train_triples_partition_path(part_no)
    for line in open(tsv_path, 'rt', encoding='utf8'):
        yield line.split("\t")


def get_train_triples_small_path():
    # about 40,000,000 lines
    tsv_path = path_join(data_path, "msmarco", "triples.train.small.tsv")
    return tsv_path


def get_msmarco_passage_collection_path():
    return path_join(data_path, "msmarco", "collection.tsv")


def get_rerank_payload_save_path(run_name):
    return path_join(output_path, "msmarco", "passage",
                     "rerank_payload", f"{run_name}.txt")


def train_triples_small_partition_iter(job_no) -> Iterator[Tuple[str, str, str]]:
    if job_no > 400:
        raise IndexError()

    # 100,000 * 400
    triplet_path = get_train_triples_small_path()
    f = open(triplet_path, "r")
    job_size = 100000  # 600 sec
    st = job_no * job_size
    ed = st + job_size
    for line in itertools.islice(f, st, ed):
        q, dp, dn = line.split("\t")
        yield q, dp, dn
