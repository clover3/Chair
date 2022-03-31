import os
import random
import subprocess
import sys
from typing import List

import psutil

from trec.trec_parse import write_trec_ranked_list_entry
from trec.types import TrecRankedListEntry


def get_ndcg_at_k(k):
    def fn(ranked_list: List[TrecRankedListEntry], true_gold: List[str]):
        pred_scores = []
        true_label = []


        # return ndcg_score([true_label], [pred_scores], k=k)
    return fn


def run_trec_eval_from_ranked_list(target_metric, qrel_path, ranked_list: List[TrecRankedListEntry]):
    trec_eval_path = os.environ["TREC_EVAL_PATH"]
    tmp_dir = os.environ["TMP_DIR"]
    name = str(random.randint(0, 1000000))
    save_path = os.path.join(tmp_dir, name)
    write_trec_ranked_list_entry(ranked_list, save_path)

    return run_trec_eval(target_metric, qrel_path, save_path, trec_eval_path)


def run_trec_eval(target_metric, qrel_path, ranked_list_path, trec_eval_path):
    if target_metric.startswith("ndcg"):
        option_metric = "ndcg_cut"
    else:
        option_metric = target_metric

    if not os.path.exists(qrel_path):
        raise FileNotFoundError(qrel_path)
    if not os.path.exists(ranked_list_path):
        raise FileNotFoundError(ranked_list_path)

    p = psutil.Popen([trec_eval_path, "-m", option_metric, qrel_path, ranked_list_path],
                     stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE,
                     )
    line_itr = iter(p.stdout.readline, b'')
    return parse_trec_eval_output(line_itr, target_metric)


def parse_trec_eval_output(line_itr, target_metric):
    lines = []
    for line in line_itr:
        line = line.decode("utf-8")
        lines.append(line)
        metric, all, score = line.split("\t")
        if metric.strip() == target_metric:
            return float(score)
    print([lines])
    raise ValueError()


def main():
    qrel_path = sys.argv[1]
    run_trec_eval_from_ranked_list(qrel_path, [])
    return


if __name__ == "__main__":
    main()
