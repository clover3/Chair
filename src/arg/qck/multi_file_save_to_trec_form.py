import json
import os
import sys
from typing import List, Dict

from arg.qck.decl import get_format_handler
from arg.qck.prediction_reader import load_combine_info_jsons
from arg.qck.save_to_trec_form import get_score_d
from arg.qck.trec_helper import scrore_d_to_trec_style_predictions
from cpath import output_path
from evals.trec import write_trec_ranked_list_entry
from misc_lib import exist_or_mkdir


def save_over_multiple_files(pred_file_list: List[str],
         info_file_path: str, run_name: str,
         input_type: str, max_entry: int, combine_strategy: str, score_type: str):
    f_handler = get_format_handler(input_type)
    info: Dict = load_combine_info_jsons(info_file_path, f_handler.get_mapping(), f_handler.drop_kdp())
    print("Info has {} entries".format(len(info)))

    score_d = {}
    for pred_file_path in pred_file_list:
        d = get_score_d(pred_file_path, info, f_handler, combine_strategy, score_type)
        score_d.update(d)
    ranked_list = scrore_d_to_trec_style_predictions(score_d, run_name, max_entry)
    save_dir = os.path.join(output_path, "ranked_list")
    exist_or_mkdir(save_dir)
    save_path = os.path.join(save_dir, run_name + ".txt")
    write_trec_ranked_list_entry(ranked_list, save_path)
    print("Saved at : ", save_path)


def multi_file_save_to_trec_form_fn(pred_file_dir: str,
         file_idx_st: int, file_idx_ed: int,
         run_config):
    pred_file_list = []
    print("multi_file_save_to_trec_form_fn")
    for idx in range(file_idx_st, file_idx_ed):
        pred_file = os.path.join(pred_file_dir, str(idx))
        pred_file_list.append(pred_file)

    not_found_file = []
    for file_path in pred_file_list:
        if not os.path.exists(file_path):
            not_found_file.append(file_path)
    ok_incomplete = 'ok_incomplete' in run_config and run_config['ok_incomplete']

    if not ok_incomplete and not_found_file:
        print("{} of {} not found".format(len(not_found_file), len(pred_file_list)))
        for e in not_found_file[:3]:
            print(e)
        return -1

    save_over_multiple_files(
        pred_file_list,
        run_config["info_path"],
        run_config["run_name"],
        run_config["input_type"],
        run_config["max_entry"],
        run_config["combine_strategy"],
        run_config["score_type"]
    )


def multi_file_save_to_trec_form_by_range(pred_file_dir: str,
         file_idx_range,
         run_config):
    pred_file_list = []
    print("multi_file_save_to_trec_form_by_range")
    for idx in file_idx_range:
        pred_file = os.path.join(pred_file_dir, str(idx))
        pred_file_list.append(pred_file)

    not_found_file = []
    for file_path in pred_file_list:
        if not os.path.exists(file_path):
            not_found_file.append(file_path)

    if not_found_file:
        print("{} of {} not found".format(len(not_found_file), len(pred_file_list)))
        for e in not_found_file[:3]:
            print(e)
        return -1

    save_over_multiple_files(
        pred_file_list,
        run_config["info_path"],
        run_config["run_name"],
        run_config["input_type"],
        run_config["max_entry"],
        run_config["combine_strategy"],
        run_config["score_type"]
    )


if __name__ == "__main__":
    run_config = json.load(open(sys.argv[1], "r"))
    multi_file_save_to_trec_form_fn(run_config["prediction_dir"],
         run_config["file_idx_st"],
         run_config["file_idx_ed"],
         run_config
    )