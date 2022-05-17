import sys

from epath import job_man_dir
from job_manager.job_runner_with_server import JobRunnerS
from trainer_v2.epr.s_bert_enc import EncodeWorker, load_segmented_data
from trainer_v2.train_util.arg_flags import flags_parser
from trainer_v2.train_util.get_tpu_strategy import get_strategy


def main(arg):
    strategy = get_strategy(args.use_tpu, args.tpu_name)
    with strategy.scope():
        # work_for_eval_split("snli", "test", arg.config_path, arg.init_checkpoint)
        work_for_eval_split("snli", "validation", arg.config_path, arg.init_checkpoint)
        # work_for_eval_split("multi_nli", "validation_matched", arg.config_path, arg.init_checkpoint)


def work_for_eval_split(dataset, split, config_path, init_checkpoint):
    job_name = f"{dataset}_{split}_sbert_align"

    def jsonl_file_load_fn(job_id):
        return load_segmented_data(dataset, split, job_id)

    def factor(output_dir):
        return EncodeWorker(output_dir, config_path, init_checkpoint, jsonl_file_load_fn)

    num_jobs = 1
    job_runner = JobRunnerS(job_man_dir, num_jobs, job_name, factor)
    job_runner.auto_runner()


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)

