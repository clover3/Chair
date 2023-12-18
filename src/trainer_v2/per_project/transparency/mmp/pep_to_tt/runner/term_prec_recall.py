import sys
from evals.basic_func import get_acc_prec_recall_i
from misc_lib import print_dict_tab
from table_lib import tsv_iter


def eval_term_pair_table(gold_d, pred_table):
    def convert_prob_score_to_label(score_s):
        score = float(score_s)
        cut = 0.1
        if score > cut:
            return 1
        else:
            return 0

    pred_list = []
    gold_list = []
    for qt, dt, score_s in pred_table:
        x = (qt, dt)
        pred = convert_prob_score_to_label(score_s)
        pred_list.append(pred)
        gold_list.append(gold_d[x])
    d = get_acc_prec_recall_i(pred_list, gold_list)
    return d


def load_fidelity_lables_from_scores(fidelity_table_path):
    fidelity_table = list(tsv_iter(fidelity_table_path))

    def convert_fidelity_score_to_label(score_s):
        score = float(score_s)
        cut = 0.1
        if score > cut:
            return 1
        elif score < -cut:
            return 0
        else:
            raise ValueError()

    gold_d = {}
    for qt, dt, score_s in fidelity_table:
        x = (qt, dt)
        label = convert_fidelity_score_to_label(score_s)
        gold_d[x] = label
    return gold_d


def main():
    fidelity_table_path = sys.argv[1]
    pred_table = list(tsv_iter(sys.argv[2]))
    gold_d = load_fidelity_lables_from_scores(fidelity_table_path)
    d = eval_term_pair_table(gold_d, pred_table)
    print_dict_tab(d)


if __name__ == "__main__":
    main()