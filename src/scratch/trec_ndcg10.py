import sys

from evals.metric_by_trec_eval import run_trec_eval
from trec.qrel_parse import load_qrels_structured
from trec.trec_parse import load_ranked_list_grouped


def main():
    metric = sys.argv[1]
    qrel_path = sys.argv[2]
    ranked_list = sys.argv[3]
    rlg = load_ranked_list_grouped(ranked_list)
    qrel = load_qrels_structured(qrel_path)
    l = []
    for qid, entries in rlg.items():
        if qid not in qrel:
            continue
        score = run_trec_eval(metric, qrel_path, entries)
        e = qid, score
        l.append(e)
        print("{}\t{}".format(qid, score))



if __name__ == "__main__":
    main()
