from collections import Counter

from arg.counter_arg_retrieval.build_dataset.path_helper import load_sliced_passage_ranked_list
from trec.qrel_parse import load_qrels_structured


def join_ranked_list_w_qrel(qrels, run_name):
    pq = load_sliced_passage_ranked_list(run_name)
    for qid, entries in pq.items():
        try:
            qrel_per_qid = qrels[qid]
            for e in entries:
                try:
                    score = qrel_per_qid[e.doc_id]
                    yield qid, e.doc_id, score
                except KeyError:
                    pass
        except KeyError:
            pass


def main():
    judgment_path = "C:\\work\\Code\\Chair\\output\\ca_building\\qrel\\0522.txt"
    qrels = load_qrels_structured(judgment_path)
    runs = ["PQ_10", "PQ_11", "PQ_12", "PQ_13"]
    rel_counter = Counter()
    for run_name in runs:
        for qid, doc_id, score in join_ranked_list_w_qrel(qrels, run_name):
            if score:
                rel_counter[(qid, doc_id)] += 1

    print("run_name, n_unique, n_rel, n_pred")
    for run_name in runs:
        n_unique = 0
        n_rel = 0
        n_pred = 0
        for qid, doc_id, score in join_ranked_list_w_qrel(qrels, run_name):
            if score:
                if rel_counter[(qid, doc_id)] == 1:
                    n_unique += 1
                n_rel += 1
            n_pred += 1

        print(run_name, n_unique, n_rel, n_pred)


if __name__ == "__main__":
    main()