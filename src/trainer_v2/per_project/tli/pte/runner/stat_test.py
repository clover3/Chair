
from scipy import stats

from typing import List

from contradiction.medical_claims.token_tagging.path_helper import get_flat_binary_save_path
from contradiction.stat_utils import read_line_scores, compute_correct
from contradiction.token_tagging.acc_eval.defs import SentTokenLabel
from dataset_specific.scientsbank.eval_fns import compute_f1_from_binary_paired
from dataset_specific.scientsbank.parse_fns import load_scientsbank_split, get_split_spec, sci_ents_test_split_list
from dataset_specific.scientsbank.pte_data_types import Question
from misc_lib import average
import numpy as np

from trainer_v2.per_project.tli.pte.path_helper import get_flat_score_save_path
from trainer_v2.per_project.tli.pte.runner.save_flat_binary_prediction import no_tune_method_list


def load_labels_flat(split):
    questions: List[Question] = load_scientsbank_split(split)
    questions.sort(key=lambda x: x.id)
    flat_labels = []
    for question in questions:
        question.student_answers.sort(key=lambda x: x.id)
        for sa in question.student_answers:
            sa.facet_entailments.sort(key=lambda x: x.facet_id)
            for fe in sa.facet_entailments:
                flat_labels.append(fe.get_bool_label())
    return flat_labels


def load_predictions(solver_name, split):
    run_name = f"{solver_name}_{split.get_save_name()}"
    if solver_name in no_tune_method_list:
        pass
    else:
        run_name = run_name + "_t"
    return read_line_scores(get_flat_score_save_path(run_name))


def get_f1(labels, preds):
    assert len(labels) == len(preds)
    binary_paired = []
    for i1, i2 in zip(labels, preds):
        binary_paired.append((bool(i1), bool(i2)))

    d = compute_f1_from_binary_paired(binary_paired)
    return d['macro_f1']


def do_bootstrap_testing(sample1, sample2, scoring_fn):
    # Compute observed test statistic
    observed_stat = scoring_fn(sample1) - scoring_fn(sample2)
    # Combine the samples
    combined = np.concatenate([sample1, sample2])
    # Number of bootstrap samples
    n_bootstrap = 1000
    # Initialize an empty array to store the bootstrap sample statistics
    bootstrap_stats = np.empty(n_bootstrap)
    # Generate bootstrap samples
    for i in range(n_bootstrap):
        bootstrap_sample = np.random.choice(combined, size=len(combined), replace=True)

        # Split the bootstrap sample into two parts
        bootstrap_sample1 = bootstrap_sample[:len(sample1)]
        bootstrap_sample2 = bootstrap_sample[len(sample1):]

        # Compute the bootstrap sample statistic
        bootstrap_stats[i] = scoring_fn(bootstrap_sample1) - scoring_fn(bootstrap_sample2)
    # Compute the p-value
    p_value = 1 - np.sum(bootstrap_stats > observed_stat) / n_bootstrap
    return p_value


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
    print('Win ratio: sys1=%.3f, sys2=%.3f, tie=%.3f' % (wins[0], wins[1], wins[2]))
    if wins[0] > wins[1]:
        print('(sys1 is superior with p value p=%.3f)' % (1 - wins[0]))
    elif wins[1] > wins[0]:
        print('(sys2 is superior with p value p=%.3f)' % (1 - wins[1]))

    # Print system stats
    sys1_scores.sort()
    sys2_scores.sort()
    print('sys1 mean=%.3f, median=%.3f, 95%% confidence interval=[%.3f, %.3f]' %
          (np.mean(sys1_scores), np.median(sys1_scores), sys1_scores[int(num_samples * 0.025)],
           sys1_scores[int(num_samples * 0.975)]))
    print('sys2 mean=%.3f, median=%.3f, 95%% confidence interval=[%.3f, %.3f]' %
          (np.mean(sys2_scores), np.median(sys2_scores), sys2_scores[int(num_samples * 0.025)],
           sys2_scores[int(num_samples * 0.975)]))
    return 1 - wins[1]


def main():
    solver_name_list = [
        "em", "w2v", "coattention", "lime", "deletion",
        "senli", "nli14", 'all_true', 'all_false', 'slr', 'gpt-3.5-turbo']
    run2 = "nlits"
    split_list = sci_ents_test_split_list
    for split_name in split_list:
        split = get_split_spec(split_name)
        labels: List[int] = load_labels_flat(split)
        def get_f1_local(preds):
            return get_f1(labels, preds)
        preds2 = load_predictions(run2, split)
        run2_score = get_f1_local(preds2)
        print("Sys2: {0} score={1:.4f}".format(run2, run2_score))
        for solver_name in solver_name_list:
            preds1 = load_predictions(solver_name, split)
            assert len(labels) == len(preds1)
            run1_score = get_f1_local(preds1)
            diff = run1_score - run2_score
            p = eval_with_paired_bootstrap(labels, preds1, preds2, get_f1,
                                           num_samples=10000, sample_ratio=0.5)
            print("Sys1: {0} score={1:.4f} diff={2:.2f} p={3:.4f} Significant={4}".format(
                solver_name, run1_score, diff, p, p < 0.01))
            print()


if __name__ == "__main__":
    main()