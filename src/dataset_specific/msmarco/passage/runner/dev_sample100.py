import random
from collections import defaultdict
from typing import List, Iterable, Callable, Dict, Tuple, Set

from dataset_specific.msmarco.passage.passage_resource_loader import load_qrel, tsv_iter
from misc_lib import path_join, TimeEstimator
from trec.types import QRelsDict



def sample_query_doc(
        top1000_iter: Iterable, qrels_dict: QRelsDict,
        n_query: int, n_doc_per_query: int,
        corpus_save_path, query_save_path):
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
        if len(entries) < 1000:
            print("Warning number of candidates smaller : {}".format(len(entries)))

        rel_doc_id = []
        try:
            for doc_id, rel in qrels_dict[qid].items():
                if rel:
                    rel_doc_id.append(doc_id)
        except KeyError as e:
            print(e)

        query = sample_qid_dict[qid]
        pos_docs = []
        neg_docs = []
        for _qid, pid, _query, text in entries:
            e = _qid, pid, _query, text
            assert _qid == qid
            assert query == _query
            if pid in rel_doc_id:
                pos_docs.append(e)
            else:
                neg_docs.append(e)

        if not pos_docs:
            print(f"Query {qid} does not have relevant docs")
            continue
        n_neg = n_doc_per_query - len(pos_docs)
        random.shuffle(neg_docs)
        target_docs = pos_docs + neg_docs[:n_neg]
        for e in target_docs:
            f_out.write("\t".join(e) + "\n")

    with open(query_save_path, "w") as f_out_q:
        for qid, query in sample_qid_dict.items():
            f_out_q.write("{}\t{}\n".format(qid, query))


def main():
    source_corpus_path = path_join("data", "msmarco", "top1000.dev")
    qrel = load_qrel("dev")
    sample_corpus_save_path = path_join("data", "msmarco", "sample_dev100", "corpus.tsv")
    sample_query_save_path = path_join("data", "msmarco", "sample_dev100", "queries.tsv")
    top1000_iter = tsv_iter(source_corpus_path)
    n_query = 100
    n_doc_per_query = 100

    sample_query_doc(top1000_iter, qrel, n_query, n_doc_per_query,
                     sample_corpus_save_path, sample_query_save_path)


if __name__ == "__main__":
    main()


