from collections import defaultdict
from typing import List, Iterable, Callable, Dict, Tuple, Set

from scipy import stats

from contradiction.medical_claims.token_tagging.acc_eval.path_helper import load_sbl_binary_label
from contradiction.medical_claims.token_tagging.path_helper import get_sbl_qrel_path, get_save_path2, \
    get_binary_save_path_w_opt
from contradiction.token_tagging.acc_eval.defs import SentTokenLabel, SentTokenBPrediction
from contradiction.token_tagging.acc_eval.parser import load_sent_token_binary_predictions
from evals.basic_func import get_acc_prec_recall, i2b
from evals.metrics import get_metric_fn
from list_lib import index_by_fn
from misc_lib import average
from runnable.trec.paired_ttest import get_score_per_query
from tab_print import print_table
from trec.qrel_parse import load_qrels_flat_per_query_0_1_only
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry
from statsmodels.stats.multitest import multipletests


def load_run(run, tag):
    prediction_path = get_save_path2(run, tag)
    ranked_list: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(prediction_path)
    return ranked_list


def paired_test_map(metric, qrels, ranked_list_1, ranked_list_2):
    metric_fn = get_metric_fn(metric)
    score_d1 = get_score_per_query(qrels, metric_fn, ranked_list_1)
    score_d2 = get_score_per_query(qrels, metric_fn, ranked_list_2)
    print("{} rankings".format(len(score_d1)))
    pairs = []
    for key in score_d1:
        try:
            e = (score_d1[key], score_d2[key])
            pairs.append(e)
        except KeyError as e:
            pass

    if len(pairs) < len(score_d1) or len(pairs) < len(score_d2):
        print("{} matched from {} and {} scores".format(len(pairs), len(score_d1), len(score_d2)))

    l1, l2 = zip(*pairs)
    d, p_value = stats.ttest_rel(l1, l2)
    print(d, p_value)
    assert d < 0
    return p_value


def do_stat_test_map(judgment_path, metric, run1_list, run2, tag):
    qrels = load_qrels_flat_per_query_0_1_only(judgment_path)
    ranked_list_2 = load_run(run2, tag)
    table = []
    for run1 in run1_list:
        try:
            ranked_list_1 = load_run(run1, tag)
            p_value = paired_test_map(metric, qrels, ranked_list_1, ranked_list_2)
            is_sig = p_value < 0.01
            table.append((run1, is_sig))
        except FileNotFoundError:
            pass
    print_table(table)


def calc_metric_per_query(label_list: List[SentTokenLabel],
                          prediction_list: List[SentTokenBPrediction],
                          metric: str) -> List[Tuple[str, float]]:
    label_d = index_by_fn(lambda x: x.qid, label_list)
    output: List[Tuple[str, float]] = []
    for p in prediction_list:
        try:
            labels = label_d[p.qid].labels
            predictions = p.predictions

            if len(labels) != len(predictions):
                print("WARNING number of tokens differ: {} != {}".format(len(labels), len(predictions)))

            per_problem: Dict = get_acc_prec_recall(i2b(predictions), i2b(labels))
            output.append((p.qid, per_problem[metric]))
        except KeyError:
            pass
    return output


def paired_test_f1(metric, labels,
                   pred1: List[SentTokenBPrediction],
                   pred2: List[SentTokenBPrediction],
                   ):
    score_d1 = dict(calc_metric_per_query(labels, pred1, metric))
    score_d2 = dict(calc_metric_per_query(labels, pred2, metric))
    print("{} rankings".format(len(score_d1)))
    pairs = []
    for key in score_d1:
        try:
            e = (score_d1[key], score_d2[key])
            pairs.append(e)
        except KeyError as e:
            pass

    if len(pairs) < len(score_d1) or len(pairs) < len(score_d2):
        print("{} matched from {} and {} scores".format(len(pairs), len(score_d1), len(score_d2)))

    l1, l2 = zip(*pairs)
    d, p_value = stats.ttest_rel(l1, l2)
    print(d, p_value)
    # assert d < 0
    return p_value


def load_bin_prediction(run_name, tag):
    save_path = get_binary_save_path_w_opt(run_name, tag, "f1")
    predictions = load_sent_token_binary_predictions(save_path)
    return predictions


def do_stat_test_f1(metric, run1_list, run2, tag):
    labels: List[SentTokenLabel] = load_sbl_binary_label(tag, "test")
    pred2 = load_bin_prediction(run2, tag)
    table = []
    p_value_list = []
    for run1 in run1_list:
        try:
            pred1 = load_bin_prediction(run1, tag)
            p_value = paired_test_f1(metric, labels, pred1, pred2)
            p_value_list.append(p_value)
            is_sig = p_value < 0.01
            table.append((run1, is_sig))
        except FileNotFoundError:
            pass
    print_table(table)

    name_for_pvalue = [f"{run1}_{tag}_{metric}" for run1 in run1_list]
    return p_value_list, name_for_pvalue


def main():
    split = "test"
    run1_list = ["exact_match", "word2vec_em",
                 "coattention", "lime", "deletion", "word_seg",
                 "davinci"
                 ]

    mismatch_only = ["exact_match", "word2vec_em", "coattention",]

    run2 = "nlits87"
    all_p_values = []
    names_all = []
    for metric in ["accuracy", "f1"]:
        for tag in ["mismatch", "conflict"]:
            if tag == "conflict":
                run1_list_selected = [run_name for run_name in run1_list if run_name not in mismatch_only]
            else:
                run1_list_selected = run1_list
            p_values, names = do_stat_test_f1(metric, run1_list_selected, run2, tag)
            all_p_values.extend(p_values)
            names_all.extend(names)


    reject, pvalue_corrected, alpha_sick, alpha_bonf = multipletests(all_p_values,
                                                                     method="bonferroni"
                                                                     )

    for name, corrected_p_value in zip(names_all, pvalue_corrected):
        print(name, corrected_p_value, corrected_p_value < 0.05)



if __name__ == "__main__":
    main()