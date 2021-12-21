from collections import Counter

from arg.counter_arg_retrieval.build_dataset.path_helper import load_sliced_passage_ranked_list
from trec.qrel_parse import load_qrels_structured


def enum_relevant_entries(qrels, run_name):
    pq = load_sliced_passage_ranked_list(run_name)

    for qid, entries in pq.items():
        qrel_per_qid = qrels[qid]
        for e in entries:
            if qrel_per_qid[e.doc_id]:
                yield qid, e.doc_id


def main():
    judgment_path = "C:\\work\\Code\\Chair\\output\\ca_building\\run4\\qrel_concat.txt"
    qrels = load_qrels_structured(judgment_path)
    runs = ["PQ_6", "PQ_7", "PQ_8", "PQ_9"]
    counter = Counter()
    for run_name in runs:
        for qid, doc_id in enum_relevant_entries(qrels, run_name):
            counter[(qid, doc_id)] += 1

    for run_name in runs:
        n_unique = 0
        n_rel = 0
        for qid, doc_id in enum_relevant_entries(qrels, run_name):
            if counter[(qid, doc_id)] == 1:
                n_unique += 1
            n_rel += 1

        print(run_name, n_unique, n_rel)


if __name__ == "__main__":
    main()