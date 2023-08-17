from cpath import output_path
from dataset_specific.msmarco.passage.passage_resource_loader import tsv_iter
from evals.basic_func import get_acc_prec_recall
from misc_lib import path_join
from tab_print import print_table


def main():
    label_path = path_join(output_path, "msmarco", "passage", "term_pair_label.txt")
    label_d = {}
    for q_term, d_term, label in tsv_iter(label_path):
        try:
            label_d[q_term, d_term] = bool(int(label))
        except ValueError:
            pass

    print("{} labels".format(len(label_d)))

    compare_save_path = path_join(output_path, "msmarco", "passage", "compare.txt")
    preds1 = []
    preds2 = []
    labels = []
    for row in tsv_iter(compare_save_path):
        q_term, d_term, score1, score2 = row
        if (q_term, d_term) in label_d:
            labels.append(label_d[q_term, d_term])
            preds1.append(float(score1) > 0)
            preds2.append(float(score2) > 0)

    metrics = ["accuracy", "precision", "recall", "f1"]
    head = metrics
    table = [head]
    for preds in [preds1, preds2]:
        d = get_acc_prec_recall(preds, labels)
        row = []
        for metric in metrics:
            row.append(d[metric])
        table.append(row)
    print_table(table)


if __name__ == "__main__":
    main()