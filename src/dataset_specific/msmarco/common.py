import csv
import gzip
import os
from typing import List, Tuple, NamedTuple

root_dir = "/mnt/nfs/work3/youngwookim/data/msmarco"
per_query_root = "/mnt/nfs/work3/youngwookim/data/msmarco/per_query"


def get_per_query_doc_path(query_id):
    save_path = os.path.join(per_query_root, str(query_id))
    return save_path


class MSMarcoDoc(NamedTuple):
    doc_id: str
    url: str
    title: str
    body: str


def load_per_query_docs(query_id):
    f = open(get_per_query_doc_path(query_id), encoding="utf-8")
    out = []
    for line in f:
        docid, url, title, body = line.split("\t")
        d = MSMarcoDoc(docid, url, title, body)
        out.append(d)
    return out


def read_queries_at(query_path) -> List[Tuple[str, str]]:
    qid_list = []
    for line in open(query_path, encoding="utf-8"):
        qid, q_text = line.split("\t")
        qid_list.append((qid, q_text))
    return qid_list


def load_train_queries() -> List[Tuple[str, str]]:
    return read_queries_at(at_working_dir("msmarco-doctrain-queries.tsv.gz"))


def load_queries(split) -> List[Tuple[str, str]]:
    return read_queries_at(at_working_dir("msmarco-doc{}-queries.tsv.gz".format(split)))


def at_working_dir(name):
    return os.path.join(root_dir, name)


def open_top100(split):
    return open(at_working_dir("msmarco-doc{}-top100".format(split)), encoding="utf-8")


def load_msmarco_qrel(qrel_path):
    qrel = {}
    with gzip.open(qrel_path, 'rt', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter=" ")
        for [topicid, _, docid, rel] in tsvreader:
            assert rel == "1"
            if topicid in qrel:
                qrel[topicid].append(docid)
            else:
                qrel[topicid] = [docid]
    return qrel


class MSMarcoDataReader:
    def __init__(self, split):
        self.split = split
        self.qrel = self.load_qrels()
        self.doc_offset = self.load_doc_offset()
        self.doc_f = open(at_working_dir("msmarco-docs.tsv"), encoding="utf8")

    def load_doc_offset(self):
        doc_offset = {}
        file_path = at_working_dir("msmarco-docs-lookup.tsv.gz")
        with gzip.open(file_path, 'rt', encoding='utf8') as f:
            tsvreader = csv.reader(f, delimiter="\t")
            for [docid, _, offset] in tsvreader:
                doc_offset[docid] = int(offset)
        return doc_offset

    def load_qrels(self):
        qrel_path = at_working_dir("msmarco-doc{}-qrels.tsv.gz".format(self.split))
        return load_msmarco_qrel(qrel_path)

    def get_content(self, docid):
        """getcontent(docid, f) will get content for a given docid (a string) from filehandle f.
        The content has four tab-separated strings: docid, url, title, body.
        """

        self.doc_f.seek(self.doc_offset[docid])
        line = self.doc_f.readline()
        assert line.startswith(docid + "\t"), \
            f"Looking for {docid}, found {line}"
        return line.rstrip()