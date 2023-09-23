import gzip

from dataset_specific.mnli.mnli_reader import NLIPairData
from dataset_specific.msmarco.passage.grouped_reader import get_train_neg5_sample_path
from dataset_specific.msmarco.passage.path_helper import get_train_triples_path
from misc_lib import TimeEstimator
from table_lib import tsv_iter
from trainer_v2.per_project.transparency.mmp.data_enum import enum_pos_neg_pointwise
from typing import List, Iterable, Callable, Dict, Tuple, Set


def iter_train_data_as_nli_pair(parition_no) -> Iterable[NLIPairData]:
    n_neg = 5
    st = parition_no
    ed = parition_no + 1
    itr = enum_pos_neg_pointwise(range(st, ed), n_neg)
    cnt = 0
    for query, doc, label in itr:
        data_id = str(parition_no) + "{0:8d}".format(cnt)
        yield NLIPairData(doc, query, str(label), data_id)
        cnt += 1


def iter_qd_sample_as_nli_pair(partition_no):
    itr = tsv_iter(get_train_neg5_sample_path(partition_no))
    cnt = 0
    for query, doc, label in itr:
        data_id = str(partition_no) + "{0:8d}".format(cnt)
        yield NLIPairData(doc, query, str(label), data_id)
        cnt += 1


def iter_train_triples_as_nli_pair(job_no=None) -> Iterable[NLIPairData]:
    cnt = 0

    def get_data_id():
        nonlocal cnt
        cnt += 1
        data_id = str(job_no) + "{0:8d}".format(cnt)
        return data_id

    tsv_path = get_train_triples_path()
    for line in gzip.open(tsv_path, 'rt', encoding='utf8'):
        query, text1, text2 = line.split("\t")
        yield NLIPairData(text1, query, "1", get_data_id())
        yield NLIPairData(text2, query, "0", get_data_id())


