from typing import Iterable, Tuple

from cache import load_pickle_from
from cpath import data_path, output_path
from dataset_specific.msmarco.passage.passage_resource_loader import FourStr, FourItem
from misc_lib import path_join, select_third_fourth
from table_lib import tsv_iter


def load_msmarco_sample_dev_as_pairs() -> Iterable[Tuple[str, str]]:
    dataset = "dev_sample100"
    return load_msmarco_sub_samples_as_qd_pair(dataset)


def load_msmarco_sub_samples_as_qd_pair(dataset):
    quad_tsv_path = get_dataset_quad_payload_path(dataset)
    tuple_itr: Iterable[Tuple[str, str]] = select_third_fourth(tsv_iter(quad_tsv_path))
    return tuple_itr


def get_dataset_quad_payload_path(dataset):
    if dataset.startswith("dev"):
        quad_tsv_path = path_join(data_path, "msmarco", dataset, "corpus.tsv")
    elif dataset == "train_when_0":
        quad_tsv_path = path_join(output_path, "msmarco", "passage", "when_full", "0")
    else:
        raise KeyError("Dataset {} is not expected".format(dataset))
    return quad_tsv_path


def get_queries_path(dataset):
    if dataset.startswith("dev"):
        tsv_path = path_join(data_path, "msmarco", dataset, "queries.tsv")
    else:
        raise KeyError("Dataset {} is not expected".format(dataset))
    return tsv_path


def enum_all_when_corpus() -> Iterable[FourStr]:
    for i in range(11):
        quad_tsv_path = path_join(output_path, "msmarco", "passage", "when_full", str(i))
        yield from tsv_iter(quad_tsv_path)


def enum_when_corpus_tokenized() -> Iterable[FourItem]:
    for i in range(17):
        save_path = path_join(output_path, "msmarco", "passage", "when_full_re_tokenized", str(i))
        yield from load_pickle_from(save_path)


def load_msmarco_sample_a_as_pairs() -> Iterable[Tuple[str, str]]:
    quad_tsv_path = path_join(data_path, "msmarco", "dev_sample1000", "corpus.tsv")
    tuple_itr: Iterable[Tuple[str, str]] = select_third_fourth(tsv_iter(quad_tsv_path))
    return tuple_itr


def get_partitioned_query_path(part_no):
    return path_join(output_path, "msmarco", "passage", "partitioned_query", str(part_no) + ".tsv")


def load_partitioned_query(part_no) -> Tuple[str, int, int]:
    f = open(get_partitioned_query_path(part_no), "r", encoding="utf-8")
    for line in f:
        q_tokens, st, ed = line.split("\t")
        yield q_tokens, int(st), int(ed)

