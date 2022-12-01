import os.path
import sys

from alignment.ists_eval.chunked_eval import chunked_solve_and_save_eval
from alignment.ists_eval.chunked_solver.nli_v3 import get_solver
from alignment.ists_eval.path_helper import get_ists_save_path
from dataset_specific.ists.split_info import ists_enum_split_genre_combs
from trainer_v2.chair_logging import c_log


def enum_prediction_todos():
    for nli_type in ["base", "pep"]:
        for label_predictor_type in ["w_context", "wo_context"]:
            for split, genre in ists_enum_split_genre_combs():
                yield {
                    'nli_type': nli_type,
                    'label_predictor_type': label_predictor_type,
                    'genre': genre,
                    'split': split,
                }


def work_todo_entry(genre, label_predictor_type, nli_type, split):
    run_name = f"{nli_type}_{label_predictor_type}"
    save_path = get_ists_save_path(genre, split, run_name)
    if os.path.exists(save_path):
        c_log.info("Skip {} {} {} {}".format(genre, label_predictor_type, nli_type, split))
        return


    solver = get_solver(nli_type, label_predictor_type)
    chunked_solve_and_save_eval(solver, run_name, genre, split)


def dev():
    work_todo_entry("images", "w_context", "base", "test")
    work_todo_entry("images", "w_context", "pep", "test")



def main():
    if len(sys.argv) > 1:
        nli_type_target = sys.argv[1]
    else:
        nli_type_target = ""
    todo_entries = enum_prediction_todos()
    for todo in todo_entries:
        nli_type: str = todo['nli_type']
        if nli_type_target and nli_type != nli_type_target:
            continue
        label_predictor_type: str = todo['label_predictor_type']
        genre = todo['genre']
        split = todo['split']
        work_todo_entry(genre, label_predictor_type, nli_type, split)


if __name__ == "__main__":
    main()