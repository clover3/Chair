import os.path
import os.path
import sys
from collections import Counter

from omegaconf import OmegaConf

from misc_lib import path_join, BinHistogram
from tab_print import tab_print_dict
from table_lib import tsv_iter
from trainer_v2.per_project.transparency.misc_common import read_lines


def main():
    case_count = [
        ("low self score", "is in top 100K frequent terms, but not ranked self"),
        ("not freq qterm", "is not in top 100K frequent terms"),
    ]

    def bin_fn(n):
        if n < 500:
            return " < 500"
        elif 500 <= n < 2000:
            return "500 <= n < 2000"
        else:
            return "n >= 2000"

    bin = BinHistogram(bin_fn)
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    q_terms = read_lines(conf.q_term_path)
    d_terms = read_lines(conf.d_term_path)
    table_save_path = conf.table_save_path
    d_terms_set = set(d_terms)
    job_size = int(conf.job_size)
    save_dir = conf.score_save_dir
    num_items = len(q_terms)
    max_job = num_items
    counter = Counter()

    out_f = open(table_save_path, "w")
    n_done = 0
    n_term_pair = 0
    for q_term_i in range(len(q_terms)):
        log_path = path_join(save_dir, f"{q_term_i}.txt")
        if not os.path.exists(log_path):
            print("not found", log_path)
            continue

        n_done += 1
        d_term_scores = [(row[0], float(row[1])) for row in tsv_iter(log_path)]
        d_term_scores_d = dict(d_term_scores)
        q_term = q_terms[q_term_i]
        q_term_not_found = "found"
        try:
            self_score = d_term_scores_d[q_term]
            threshold = self_score
        except KeyError as e:
            if q_term in d_terms_set:
                q_term_not_found = "low self score"
                counter["low self score"] += 1
            else:
                q_term_not_found = "not freq qterm"
                counter["not freq qterm"] += 1
            threshold = 2

        selected_terms = [d_term for d_term, score in d_term_scores if score >= threshold]
        try:
            selected_terms.remove(q_term)
        except ValueError:
            pass

        key = bin_fn(len(selected_terms))
        counter[q_term_not_found, key] += 1

        bin.add(len(selected_terms))
        n_term_pair += len(selected_terms)

    tab_print_dict(counter)
    print("n_term_pair", n_term_pair)
    print("n_done", n_done)


if __name__ == "__main__":
    main()
