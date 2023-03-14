import csv
import gzip
from typing import Iterable, Tuple

from cpath import at_data_dir
from misc_lib import path_join, select_third_fourth
from trec.qrel_parse import load_qrels_structured
from trec.types import QRelsDict


def tsv_iter(file_path) -> Iterable[Tuple]:
    f = open(file_path, "r", encoding="utf-8", errors="ignore")
    reader = csv.reader(f, delimiter='\t')
    return reader


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
    file_path = at_data_dir("msmarco", "queries.{}.tsv".format(split))
    f = open(file_path, "r", encoding="utf-8", errors="ignore")
    reader = csv.reader(f, delimiter='\t')

    output = {}
    for idx, row in enumerate(reader):
        qid, q_text = row
        output[qid] = q_text
    return output


def load_msmarco_sample_dev_as_pairs():
    quad_tsv_path = path_join("data", "msmarco", "sample_dev", "corpus.tsv")
    tuple_itr: Iterable[Tuple[str, str]] = select_third_fourth(tsv_iter(quad_tsv_path))
    return tuple_itr