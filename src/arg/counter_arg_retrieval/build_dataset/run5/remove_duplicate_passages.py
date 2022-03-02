# Remove duplicate passages
import os

from arg.counter_arg_retrieval.build_dataset.data_prep.remove_duplicate_passages import SplitDocDict, \
    remove_duplicate_passages
from arg.counter_arg_retrieval.build_dataset.run5.passage_scoring_util import load_run5_swtt_passage_as_d
from cpath import output_path
from list_lib import flatten
from misc_lib import get_dir_files
from trec.trec_parse import load_ranked_list_grouped, write_trec_ranked_list_entry


def do_for_query(query_id, rlg_path, new_rlg_save_path):
    doc_as_passage_dict: SplitDocDict = load_run5_swtt_passage_as_d(query_id)
    rlg = load_ranked_list_grouped(rlg_path)
    new_rlg = remove_duplicate_passages(rlg, doc_as_passage_dict)

    write_trec_ranked_list_entry(flatten(new_rlg.values()), new_rlg_save_path)


def do_for_run(run_name):
    old_rlg_dir = os.path.join(output_path, "ca_building", "run4", "passage_ranked_list_100", run_name)
    new_rlg_dir = os.path.join(output_path, "ca_building", "run4", "passage_ranked_list_100_unique", run_name)

    os.makedirs(new_rlg_dir, exist_ok=True)
    n_files_expected = 53
    file_list = list(get_dir_files(old_rlg_dir))
    if len(file_list) != n_files_expected:
        print("Expected {} but {} files found ".format(n_files_expected, len(file_list)))

    for file_path in file_list:
        query_id = os.path.basename(file_path)
        rlg_path = os.path.join(old_rlg_dir, query_id)
        new_rlg_save_path = os.path.join(new_rlg_dir, query_id)
        do_for_query(query_id, rlg_path, new_rlg_save_path)


def main():
    run_name = NotImplemented
    do_for_run(run_name)


if __name__ == "__main__":
    main()