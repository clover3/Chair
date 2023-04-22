import csv
import gzip
import random
from typing import List, Iterable, Callable, Dict, Tuple, Set, Any

from cache import load_pickle_from
from cpath import at_data_dir, data_path, output_path
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


def load_msmarco_sample_dev_as_pairs() -> Iterable[Tuple[str, str]]:
    dataset = "dev_sample100"
    return load_msmarco_sub_samples(dataset)


def load_msmarco_sub_samples(dataset):
    if dataset.startswith("dev"):
        quad_tsv_path = path_join(data_path, "msmarco", dataset, "corpus.tsv")
    elif dataset == "train_when_0":
        quad_tsv_path = path_join(output_path, "msmarco", "passage", "when_full", "0")
    else:
        raise KeyError("Dataset {} is not expected".format(dataset))
    tuple_itr: Iterable[Tuple[str, str]] = select_third_fourth(tsv_iter(quad_tsv_path))
    return tuple_itr


FourStr = Tuple[str, str, str, str]
FourItem = Tuple[str, str, Any, Any]


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


def load_msmarco_collection()-> Iterable[Tuple[str, str]]:
    tsv_path = path_join(data_path, "msmarco", "collection.tsv")
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


class MMPPosNegSampler:
    def __init__(self):
        self.qrel = load_qrel("train")

    def split_pos_neg_entries(self, entries, qid=None):
        if qid is None:
            qid = entries[0][0]

        pos_doc_ids = []
        for doc_id, score in self.qrel[qid].items():
            if score > 0:
                pos_doc_ids.append(doc_id)

        pos_doc = []
        neg_doc = []
        for e in entries:
            qid, pid, query, text = e
            if pid in pos_doc_ids:
                pos_doc.append(e)
            else:
                neg_doc.append(e)
        return pos_doc, neg_doc

    def sample_pos_neg(self, group):
        pos_docs, neg_docs = self.split_pos_neg_entries(group)
        if len(neg_docs) == 0:
            raise IndexError
        neg_idx = random.randrange(len(neg_docs))
        return pos_docs[0], neg_docs[neg_idx]
