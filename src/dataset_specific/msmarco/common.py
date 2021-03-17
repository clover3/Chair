import csv
import gzip
import os
from collections import defaultdict
from typing import Dict
from typing import List, Tuple, NamedTuple, NewType

from cache import load_pickle_from
from epath import job_man_dir
from list_lib import left, lmap
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry

root_dir = "/mnt/nfs/work3/youngwookim/data/msmarco"
per_query_root = "/mnt/nfs/work3/youngwookim/data/msmarco/per_query"


def get_per_query_doc_path(query_id):
    save_path = os.path.join(per_query_root, str(query_id))
    return save_path


QueryID = NewType('QueryID', str)


class MSMarcoDoc(NamedTuple):
    doc_id: str
    url: str
    title: str
    body: str


def load_per_query_docs(query_id, empty_doc_fn) -> List[MSMarcoDoc]:
    f = open(get_per_query_doc_path(query_id), encoding="utf-8")
    out = []
    for idx, line in enumerate(f):
        try:
            docid, url, title, body = line.split("\t")
            d = MSMarcoDoc(docid, url, title, body)
            out.append(d)
        except ValueError:
            tokens = line.strip().split("\t")
            if len(tokens) > 1 and tokens[1].startswith("http"):
                doc_id = tokens[0]
                url = tokens[1]
                title = ""
                body = ""
                d = MSMarcoDoc(doc_id, url, title, body)
                out.append(d)
    return out


def load_candidate_doc_list_1(split) -> Dict[QueryID, List[str]]:
    save_path = os.path.join(root_dir, "target_doc_{}.tsv".format(split))
    return load_candidate_doc_list(save_path)


def load_candidate_doc_list_10(split) -> Dict[QueryID, List[str]]:
    save_path = os.path.join(root_dir, "train_docs_10times_{}.tsv".format(split))
    return load_candidate_doc_list(save_path)


def load_candidate_doc_top50(split) -> Dict[QueryID, List[str]]:
    save_path = os.path.join(root_dir, "train_docs_top50_{}.tsv".format(split))
    return load_candidate_doc_list(save_path)




def load_candidate_doc_list(save_path):
    output = {}
    for line in open(save_path, "r"):
        tokens = line.strip().split("\t")
        query_id = QueryID(tokens[0])
        doc_ids = tokens[1:]
        output[query_id] = doc_ids
    return output


def read_queries_at(query_path) -> List[Tuple[QueryID, str]]:
    qid_list = []
    # for line in open(query_path, encoding="utf-8"):
    for line in gzip.open(query_path, 'rt', encoding='utf8'):
        qid, q_text = line.split("\t")
        qid_list.append((QueryID(qid), q_text))
    return qid_list


def load_train_queries() -> List[Tuple[QueryID, str]]:
    return read_queries_at(at_working_dir("msmarco-doctrain-queries.tsv.gz"))


def load_queries(split) -> List[Tuple[QueryID, str]]:
    return read_queries_at(at_working_dir("msmarco-doc{}-queries.tsv.gz".format(split)))


def load_query_group(split) -> List[List[QueryID]]:
    qids = left(load_queries(split))
    if split == "train":
        n_per_group = 1000
    else:
        n_per_group = 100
    qids.sort()
    st = 0
    group_list = []
    while st < len(qids):
        ed = st + n_per_group
        group_list.append(qids[st:ed])
        st = ed
    return group_list


def at_working_dir(name):
    return os.path.join(root_dir, name)


def open_top100(split):
    return open(at_working_dir("msmarco-doc{}-top100".format(split)), encoding="utf-8")


def load_token_d_1(split, query_id) -> Dict[str, List[str]]:
    save_path = os.path.join(job_man_dir, "MSMARCO_{}_tokens".format(split), query_id)
    return load_pickle_from(save_path)


def load_token_d_10doc(split, query_id) -> Dict[str, List[str]]:
    save_path = os.path.join(job_man_dir, "MSMARCO_{}_doc10_tokens".format(split), query_id)
    return load_pickle_from(save_path)


def load_token_d_50doc(split, query_id) -> Dict[str, List[str]]:
    save_path = os.path.join(job_man_dir, "MSMARCO_{}_top50_tokens".format(split), query_id)
    return load_pickle_from(save_path)



def load_msmarco_qrel_from_gz(qrel_path) -> Dict[QueryID, List[str]]:
    qrel = {}
    with gzip.open(qrel_path, 'rt', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter=" ")
        for [topicid, _, docid, rel] in tsvreader:
            topicid = QueryID(topicid)
            assert rel == "1"
            if topicid in qrel:
                qrel[topicid].append(docid)
            else:
                qrel[topicid] = [docid]
    return qrel


def load_msmarco_qrel_from_txt(qrel_path) -> Dict[QueryID, List[str]]:
    qrel = defaultdict(list)
    with open(qrel_path, 'r', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter=" ")
        for [topicid, _, docid, rel] in tsvreader:
            topicid = QueryID(topicid)
            if int(rel) > 0:
                qrel[topicid].append(docid)
    return qrel


class SimpleQrel:
    def __init__(self, qrel_d: Dict[QueryID, List[str]]):
        self.qrel_d = qrel_d

    def get_label(self, qid, doc_id):
        if qid in self.qrel_d:
            if doc_id in self.qrel_d[qid]:
                return True
            else:
                return False
        else:
            return False


def load_msmarco_raw_qrels(split) -> Dict[QueryID, List[str]]:
    if split == "test":
        qrel_path = at_working_dir("msmarco-doc{}-qrels.tsv".format(split))
        return load_msmarco_qrel_from_txt(qrel_path)
    else:
        qrel_path = at_working_dir("msmarco-doc{}-qrels.tsv.gz".format(split))
        return load_msmarco_qrel_from_gz(qrel_path)


def load_msmarco_simple_qrels(split) -> SimpleQrel:
    if split == "test":
        qrel_path = at_working_dir("msmarco-doc{}-qrels.tsv".format(split))
        raw_qrel = load_msmarco_qrel_from_txt(qrel_path)
    else:
        qrel_path = at_working_dir("msmarco-doc{}-qrels.tsv.gz".format(split))
        raw_qrel = load_msmarco_qrel_from_gz(qrel_path)
    return SimpleQrel(raw_qrel)


def top100_doc_ids(split) -> Dict[QueryID, List[str]]:
    rlg: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(at_working_dir("msmarco-doc{}-top100".format(split)))
    out_d = {}
    for qid, entries in rlg.items():
        doc_ids = lmap(TrecRankedListEntry.get_doc_id, entries)
        out_d[QueryID(qid)] = doc_ids
    return out_d


class MSMarcoDataReader:
    def __init__(self, split):
        self.split = split
        self.qrel = load_msmarco_raw_qrels(split)
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


    def get_content(self, docid):
        """getcontent(docid, f) will get content for a given docid (a string) from filehandle f.
        The content has four tab-separated strings: docid, url, title, body.
        """

        self.doc_f.seek(self.doc_offset[docid])
        line = self.doc_f.readline()
        assert line.startswith(docid + "\t"), \
            f"Looking for {docid}, found {line}"
        return line.rstrip()