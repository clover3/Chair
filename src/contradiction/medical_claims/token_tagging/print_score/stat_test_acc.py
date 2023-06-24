from scipy import stats

from contradiction.medical_claims.token_tagging.acc_eval.path_helper import load_sbl_binary_label
from contradiction.medical_claims.token_tagging.path_helper import get_flat_binary_save_path, \
    mismatch_only_method_list
from typing import List

from contradiction.stat_utils import read_line_scores, compute_correct
from contradiction.token_tagging.acc_eval.defs import SentTokenLabel
from misc_lib import average
import numpy as np


def do_bootstrap_testing(sample1, sample2):
    # Compute observed test statistic
    observed_stat = np.mean(sample1) - np.mean(sample2)
    # Combine the samples
    combined = np.concatenate([sample1, sample2])
    # Number of bootstrap samples
    n_bootstrap = 10000
    # Initialize an empty array to store the bootstrap sample statistics
    bootstrap_stats = np.empty(n_bootstrap)
    # Generate bootstrap samples
    for i in range(n_bootstrap):
        bootstrap_sample = np.random.choice(combined, size=len(combined), replace=True)

        # Split the bootstrap sample into two parts
        bootstrap_sample1 = bootstrap_sample[:len(sample1)]
        bootstrap_sample2 = bootstrap_sample[len(sample1):]

        # Compute the bootstrap sample statistic
        bootstrap_stats[i] = np.mean(bootstrap_sample1) - np.mean(bootstrap_sample2)
    # Compute the p-value
    p_value = np.sum(bootstrap_stats > observed_stat) / n_bootstrap
    return p_value


def load_labels_flat(tag):
    labels: List[SentTokenLabel] = load_sbl_binary_label(tag, "test")
    labels.sort(key=lambda x: x.qid)
    label_flat_list = []
    for per_sent in labels:
        label_flat_list.extend(per_sent.labels)
    return label_flat_list


def load_scores(run_name, tag, metric):
    return read_line_scores(get_flat_binary_save_path(run_name, tag, metric))


def main():
    metric = "accuracy"
    for tag in ["mismatch", "conflict"]:
        print(tag)
        labels: List[int] = load_labels_flat(tag)
        run_list = ["exact_match", "word2vec_em", "coattention",
                    "lime", "deletion", "senli", "word_seg",
                    "slr",
                     "davinci", 'gpt-3.5-turbo'
                     ]
        run2 = "nlits87"
        scores2 = load_scores(run2, tag, metric)
        cl2 = compute_correct(labels, scores2)
        print("{0} average={1:.4f}".format(run2, average(cl2)))
        assert len(labels) == len(scores2)

        for run1 in run_list:
            if tag == "conflict" and run1 in mismatch_only_method_list:
                continue
            scores1 = load_scores(run1, tag, metric)
            assert len(labels) == len(scores1)
            cl1 = compute_correct(labels, scores1)

            diff, p = stats.ttest_rel(cl1, cl2)
            # Compute the observed test statistic
            # p_value = do_bootstrap_testing(cl1, cl2)
            print("{0} average={1:.4f} p={2} {3}".format(
                run1, average(cl1), p, p < 0.01))


if __name__ == "__main__":
    main()