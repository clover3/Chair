# Remove duplicate passages
import os
import sys

from arg.counter_arg_retrieval.build_dataset.data_prep.remove_duplicate_passages import SplitDocDict, \
    remove_duplicate_passages
from arg.counter_arg_retrieval.build_dataset.run5.passage_scoring_util import load_run5_swtt_passage_as_d
from cpath import output_path
from list_lib import flatten
from misc_lib import TEL
from trec.trec_parse import load_ranked_list_grouped, write_trec_ranked_list_entry


def do_for_query(rlg_path, new_rlg_save_path):
    rlg = load_ranked_list_grouped(rlg_path)
    qid_list = list(rlg.keys())
    new_rlg = {}
    for qid in TEL(qid_list):
        print(qid)
        items = rlg[qid]
        doc_as_passage_dict: SplitDocDict = load_run5_swtt_passage_as_d(qid)
        rlg_per_query = {qid: items}
        new_rlg_per_query = remove_duplicate_passages(rlg_per_query, doc_as_passage_dict)
        new_rlg[qid] = new_rlg_per_query[qid]
    write_trec_ranked_list_entry(flatten(new_rlg.values()), new_rlg_save_path)


def do_for_run(run_name):
    file_name = "{}.txt".format(run_name)
    in_dir_name = "passage_ranked_list_100"
    old_rlg = os.path.join(output_path, "ca_building", in_dir_name, file_name)
    out_dir_name = "passage_ranked_list_100_unique"
    new_rlg = os.path.join(output_path, "ca_building", out_dir_name, file_name)
    do_for_query(old_rlg, new_rlg)


def main():
    run_name = sys.argv[1]
    do_for_run(run_name)


if __name__ == "__main__":
    main()