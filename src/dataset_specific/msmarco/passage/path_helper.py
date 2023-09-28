from cpath import data_path
from misc_lib import path_join
from table_lib import tsv_iter
from typing import List, Iterable, Callable, Dict, Tuple, Set


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


def get_mmp_test_qrel_json(dataset_name):
    return path_join(data_path, "msmarco", "passage", dataset_name, "qrel_binary.json")


# 397,756,691

def get_train_triples_path():
    tsv_path = path_join(data_path, "msmarco", "triples.train.full.tsv.gz")
    return tsv_path


def get_train_triples_small_path():
    tsv_path = path_join(data_path, "msmarco", "triples.train.small.tsv")
    return tsv_path


def get_msmarco_passage_collection_path():
    return path_join(data_path, "msmarco", "collection.tsv")