import argparse
import sys
from adhoc.eval_helper.line_format_to_trec_ranked_list import \
    build_ranked_list_from_qid_pid_scores

# Build ranked list by combining scores and qid_pid' assume they are paired
arg_parser = argparse.ArgumentParser(description='')
arg_parser.add_argument("--scores_path",)
arg_parser.add_argument("--qid_pid_path" )
arg_parser.add_argument("--save_path")
arg_parser.add_argument("--run_name")


#
# def dev_splade():
#     run_name = "splade"
#     scores_path = at_output_dir("lines_scores", "splade_dev_sample.txt")
#     qid_pid_path = path_join("data", "msmarco", "sample_dev100", "corpus.tsv")
#     qid_pid: List[Tuple[str, str]] = list(select_first_second(tsv_iter(qid_pid_path)))
#     save_path = at_output_dir("ranked_list", "splade_mmp_dev_sample.txt")
#     build_ranked_list_from_qid_pid_scores(qid_pid, run_name, save_path, scores_path)
#

def main(args):
    build_ranked_list_from_qid_pid_scores(
        args.qid_pid_path,
        args.run_name,
        args.save_path,
        args.scores_path)


if __name__ == "__main__":
    args = arg_parser.parse_args(sys.argv[1:])
    main(args)
    #

