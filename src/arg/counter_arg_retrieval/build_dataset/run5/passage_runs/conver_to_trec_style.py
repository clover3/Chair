import os

from arg.counter_arg_retrieval.build_dataset.passage_prediction_summarizer import convert_json_prediction_to_trec_style
from cpath import output_path
from misc_lib import get_dir_files, tprint


def enum_files_for_runs(run_name):
    save_dir = os.path.join(output_path, "ca_building", "run5", "passage_predictions", run_name)

    for file_path in get_dir_files(save_dir):
        file_name = os.path.basename(file_path)
        yield file_path, file_name


def do_for_run(run_name):
    save_dir = os.path.join(output_path, "ca_building", "run5", "passage_ranked_list_per_query", run_name)
    os.makedirs(save_dir, exist_ok=True)

    for json_file_path, file_name in enum_files_for_runs(run_name):
        tprint(file_name)
        save_path = os.path.join(save_dir, file_name)
        convert_json_prediction_to_trec_style(json_file_path, run_name, save_path)


def main():
    for run_name in ["PQ_11", "PQ_12"]:
        do_for_run(run_name)


if __name__ == "__main__":
    main()