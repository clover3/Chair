import random
from collections import defaultdict
from typing import List, Iterable, Callable, Dict, Tuple, Set

from dataset_specific.msmarco.passage.passage_resource_loader import load_qrel
from table_lib import tsv_iter
from misc_lib import path_join, TimeEstimator
from trec.types import QRelsDict


def sample_query_doc(
        top1000_iter: Iterable,
        n_query: int,
        corpus_save_path,
        query_save_path):
    max_doc_per_query = 1000

    f_out = open(corpus_save_path, "w")

    sample_qid_dict = {}
    qid_clustered = defaultdict(list)

    for qid, pid, query, text in top1000_iter:
        e = qid, pid, query, text
        if len(sample_qid_dict) < n_query:
            sample_qid_dict[qid] = query

        if qid in sample_qid_dict:
            qid_clustered[qid].append(e)

    # validate
    for qid, entries in qid_clustered.items():
        if len(entries) < max_doc_per_query:
            print("Warning number of candidates smaller : {}".format(len(entries)))

        query = sample_qid_dict[qid]
        for _qid, pid, _query, text in entries:
            assert _qid == qid
            assert query == _query

        for e in entries:
            f_out.write("\t".join(e) + "\n")

        if len(entries) != max_doc_per_query:
            print("It has {} docs".format(len(entries)))

    with open(query_save_path, "w") as f_out_q:
        for qid, query in sample_qid_dict.items():
            f_out_q.write("{}\t{}\n".format(qid, query))


def main():
    source_corpus_path = path_join("data", "msmarco", "top1000.dev")
    sample_corpus_save_path = path_join("data", "msmarco", "sample_dev1000", "corpus.tsv")
    sample_query_save_path = path_join("data", "msmarco", "sample_dev1000", "queries.tsv")
    top1000_iter = tsv_iter(source_corpus_path)
    n_query = 1000

    sample_query_doc(top1000_iter, n_query,
                     sample_corpus_save_path, sample_query_save_path)


if __name__ == "__main__":
    main()


