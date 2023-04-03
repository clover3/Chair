import random
from collections import defaultdict
from typing import List, Iterable, Callable, Dict, Tuple, Set

from dataset_specific.msmarco.passage.passage_resource_loader import load_qrel, tsv_iter
from misc_lib import path_join, TimeEstimator
from trec.qrel_parse import load_qrels_structured


def sample_query_doc(
        top1000_iter: Iterable,
        n_query: int,
        corpus_save_path,
        qrel_path):
    f_out = open(corpus_save_path, "w")
    qrels = load_qrels_structured(qrel_path)

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
        qrel_e = qrels[qid]
        if len(entries) < 1000:
            print("Warning number of candidates smaller : {}".format(len(entries)))

        pos_doc = []
        neg_doc = []
        query = sample_qid_dict[qid]
        for _qid, pid, _query, text in entries:
            if pid in qrel_e and qrel_e[pid]:
                pos_doc.append((pid, text))
            else:
                neg_doc.append((pid, text))

            assert _qid == qid
            assert query == _query

        def write_entry(e):
            f_out.write("\t".join(e) + "\n")

        for e in pos_doc:
            score = 1
            pid, text = e
            write_entry((qid, pid, query, text, score))

        neg_selected = neg_doc[:5]
        random.shuffle(neg_doc[10:])
        neg_selected.extend(neg_doc[:5])

        for e in neg_selected:
            score = 0
            pid, text = e
            write_entry((qid, pid, query, text, score))




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


