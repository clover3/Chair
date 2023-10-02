import csv
import gzip
import random
from typing import List, Iterable, Tuple, Any

from cache import load_pickle_from
from cpath import at_data_dir, data_path, output_path
from dataset_specific.msmarco.passage.path_helper import get_msmarco_passage_collection_path
from misc_lib import path_join, select_third_fourth
from table_lib import tsv_iter
from trec.qrel_parse import load_qrels_structured
from trec.types import QRelsDict


FourStr = Tuple[str, str, str, str]
FourItem = Tuple[str, str, Any, Any]


def enumerate_triple():
    gz_path = at_data_dir("msmarco", "qidpidtriples.train.full.2.tsv.gz")
    for line in gzip.open(gz_path, 'rt', encoding='utf8'):
        qid, pid1, pid2 = line.split("\t")
        yield qid, pid1, pid2


def load_qrel(split) -> QRelsDict:
    msmarco_passage_qrel_path = at_data_dir("msmarco", "qrels.{}.tsv".format(split))
    passage_qrels: QRelsDict = load_qrels_structured(msmarco_passage_qrel_path)
    return passage_qrels


def load_queries_as_d(split):
    reader = enum_queries(split)
    output = {}
    for idx, row in enumerate(reader):
        qid, q_text = row
        output[qid] = q_text
    return output


def enum_queries(split):
    file_path = at_data_dir("msmarco", "queries.{}.tsv".format(split))
    f = open(file_path, "r", encoding="utf-8", errors="ignore")
    reader = csv.reader(f, delimiter='\t')
    return reader


def load_msmarco_collection() -> Iterable[Tuple[str, str]]:
    tsv_path = get_msmarco_passage_collection_path()
    return tsv_iter(tsv_path)


def enum_grouped(iter: Iterable[FourItem]) -> Iterable[List[FourItem]]:
    prev_qid = None
    group = []
    for qid, pid, query, text in iter:
        if prev_qid is not None and prev_qid != qid:
            yield group
            group = []

        prev_qid = qid
        group.append((qid, pid, query, text))

    yield group


def enum_grouped2(iter: Iterable[Tuple]) -> Iterable[List[Tuple]]:
    prev_qid = None
    group = []
    for e in iter:
        group_id = e[0]
        if prev_qid is not None and prev_qid != group_id:
            yield group
            group = []

        prev_qid = group_id
        group.append(e)

    yield group


class MMPPosNegSampler(PosNegSampler):
    def __init__(self):
        qrel = load_qrel("train")
        super(MMPPosNegSampler, self).__init__(qrel)
