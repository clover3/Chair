import sys

from cpath import output_path
from misc_lib import path_join
from tab_print import print_table
from table_lib import tsv_iter
from trainer_v2.per_project.transparency.mmp.pep_to_tt.runner.term_prec_recall import load_fidelity_lables_from_scores, \
    eval_term_pair_table


def main():
    fidelity_table_path = path_join(output_path, "msmarco", "passage", "fidelity_2_2_pos_neg.tsv")
    gold_d = load_fidelity_lables_from_scores(fidelity_table_path)

    pred_table_path_format = path_join(
        output_path, "mmp", "tables", "fidelity2_2_tt3_{}.tsv")
    print(pred_table_path_format)

    columns = ['accuracy', 'precision', 'recall', 'f1']
    table1 = [["steps"] + columns]
    for step in range(10000, 90001, 10000):
        pred_table_path = pred_table_path_format.format(step)
        row = [str(step)]
        try:
            pred_table = list(tsv_iter(pred_table_path))
            d = eval_term_pair_table(gold_d, pred_table)
            row.extend([d[c] for c in columns])
        except FileNotFoundError:
            row.extend(["-" for c in columns])
        table.append(row)

    print_table(table)


if __name__ == "__main__":
    main()
