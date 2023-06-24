
from scipy import stats

from typing import List

from contradiction.medical_claims.token_tagging.acc_eval.path_helper import load_sbl_binary_label
from contradiction.medical_claims.token_tagging.path_helper import get_flat_binary_save_path, mismatch_only_method_list
from contradiction.medical_claims.token_tagging.print_score.stat_test_acc import load_labels_flat
from contradiction.stat_utils import read_line_scores, compute_correct
from contradiction.token_tagging.acc_eval.defs import SentTokenLabel
from evals.basic_func import get_acc_prec_recall, i2b
from misc_lib import average
import numpy as np


def load_scores(run_name, tag, metric) -> List[int]:
    return read_line_scores(get_flat_binary_save_path(run_name, tag, metric))


def get_f1(labels: List[int], preds: List[int]):
    assert len(labels) == len(preds)
    acc_prec_recall = get_acc_prec_recall(i2b(preds), i2b(labels))
    return acc_prec_recall['f1']


def eval_with_paired_bootstrap(gold, sys1, sys2, eval_measure,
                               num_samples=10000, sample_ratio=0.5,
                               ):
    ''' Evaluate with paired boostrap
    This compares two systems, performing a significance tests with
    paired bootstrap resampling to compare the accuracy of the two systems.

    :param gold: The correct labels
    :param sys1: The output of system 1
    :param sys2: The output of system 2
    :param num_samples: The number of bootstrap samples to take
    :param sample_ratio: The ratio of samples to take every time
    :param eval_type: The type of evaluation to do (acc, pearson, bleu, bleu_detok)
    '''
    assert (len(gold) == len(sys1))
    assert (len(gold) == len(sys2))

    # Preprocess the data appropriately for they type of eval
    sys1_scores = []
    sys2_scores = []
    wins = [0, 0, 0]
    n = len(gold)
    ids = list(range(n))

    for _ in range(num_samples):
        # Subsample the gold and system outputs
        reduced_ids = np.random.choice(ids, int(len(ids) * sample_ratio), replace=True)
        reduced_gold = [gold[i] for i in reduced_ids]
        reduced_sys1 = [sys1[i] for i in reduced_ids]
        reduced_sys2 = [sys2[i] for i in reduced_ids]
        # Calculate accuracy on the reduced sample and save stats
        sys1_score = eval_measure(reduced_gold, reduced_sys1)
        sys2_score = eval_measure(reduced_gold, reduced_sys2)
        if sys1_score > sys2_score:
            wins[0] += 1
        elif sys1_score < sys2_score:
            wins[1] += 1
        else:
            wins[2] += 1
        sys1_scores.append(sys1_score)
        sys2_scores.append(sys2_score)

    # Print win stats
    wins = [x / float(num_samples) for x in wins]
    # Print system stats
    sys1_scores.sort()
    sys2_scores.sort()
    return 1 - wins[1]


def main():
    run_list = [
                "exact_match",
                "word2vec_em",
                "coattention",
                "lime",
                "deletion",
                "senli",
                "slr",
                "word_seg",
                "davinci",
                "gpt-3.5-turbo"
                ]
    run2 = "nlits87"
    metric = "f1"
    tag_list = ["mismatch", "conflict"]
    for tag in tag_list:
        labels: List[int] = load_labels_flat(tag)

        def get_f1_local(preds):
            return get_f1(labels, preds)

        preds2 = load_scores(run2, tag, metric)
        assert len(labels) == len(preds2)
        run2_score = get_f1_local(preds2)
        print("Sys2: {0} score={1:.4f}".format(run2, run2_score))
        for run1 in run_list:
            if tag == "conflict" and run1 in mismatch_only_method_list:
                continue

            preds1 = load_scores(run1, tag, metric)

            assert len(labels) == len(preds1)
            run1_score = get_f1_local(preds1)
            diff = run1_score - run2_score
            p = eval_with_paired_bootstrap(labels, preds1, preds2, get_f1,
                                           num_samples=10000, sample_ratio=0.5)
            print("Sys1: {0} score={1:.4f} diff={2:.2f} p={3:.4f} Significant={4}".format(
                run1, run1_score, diff, p, p < 0.01))


if __name__ == "__main__":
    main()