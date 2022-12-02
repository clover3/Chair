import os

from alignment.ists_eval.eval_analysis_f1_calc import calc_type_specific_f1, calc_confucion, print_failure_outer
from alignment.ists_eval.matrix_eval_helper import load_ists_predictions
from alignment.ists_eval.path_helper import get_ists_save_path
from dataset_specific.ists.parse import AlignmentPredictionList, type_list, ALIGN_EQUI, ALIGN_SPE1, ALIGN_SPE2
from dataset_specific.ists.path_helper import load_ists_label, load_ists_problems_w_chunk
from dataset_specific.ists.split_info import ists_enum_split_genre_combs
from tab_print import print_table, concat_table_horizontally


def enum_run_name():
    for nli_type in ["base", "pep"]:
        for label_predictor_type in ["w_context", "wo_context"]:
            run_name = f"{nli_type}_{label_predictor_type}"
            yield run_name

def main_2():
    type_list_to_print = [ALIGN_EQUI, ALIGN_SPE1, ALIGN_SPE2]


    head1 = ["dataset", "type"]
    for run_name in enum_run_name():
        head1.append(run_name)
        head1.append("")
        head1.append("")

    head2 = ["", ""]
    for run_name in enum_run_name():
        head2.extend(["prec", "recall", "f1"])
    table = [head1, head2]

    for split, genre in ists_enum_split_genre_combs():
        gold: AlignmentPredictionList = load_ists_label(genre, split)
        for target_type in type_list_to_print:
            row = ["{}_{}".format(genre, split), target_type]
            for run_name in enum_run_name():
                save_path = get_ists_save_path(genre, split, run_name)
                if os.path.exists(save_path):
                    pred: AlignmentPredictionList = load_ists_predictions(genre, split, run_name)
                    scores = calc_type_specific_f1(gold, pred, target_type)
                    s = map("{0:.3f}".format, [scores["precision"], scores["recall"], scores["f1"]])
                    row.extend(s)
                else:
                    row.extend(["-","-","-",])
            table.append(row)
    print_table(table)


def do_print_confusion():
    run_name_list = ["base_wo_context", "pep_w_context"]

    def make_confusion_table(counter):
        head = [""] + type_list
        table = [head]
        for type_1 in type_list:
            row = [type_1]
            for type_2 in type_list:
                row.append(counter[(type_1, type_2)])
            table.append(row)
        return table

    for split, genre in ists_enum_split_genre_combs():
        gold: AlignmentPredictionList = load_ists_label(genre, split)
        print(split, genre)
        for run_name in run_name_list:
            pred: AlignmentPredictionList = load_ists_predictions(genre, split, run_name)
            conf_pg, conf_gp = calc_confucion(gold, pred)
            print(run_name)
            left = make_confusion_table(conf_pg)
            right = make_confusion_table(conf_gp)

            table = concat_table_horizontally([left, right])
            print_table(table)


def show_error():
    run_name1 = "pep_w_context"
    run_name2 = "base_wo_context"
    for split, genre in [("train", "images"), ("train", "headlines")]:
        problems = load_ists_problems_w_chunk(genre, split)
        gold: AlignmentPredictionList = load_ists_label(genre, split)
        pred: AlignmentPredictionList = load_ists_predictions(genre, split, run_name1)
        pred2: AlignmentPredictionList = load_ists_predictions(genre, split, run_name2)
        print_failure_outer(gold, pred, pred2, problems)


if __name__ == "__main__":
    show_error()
