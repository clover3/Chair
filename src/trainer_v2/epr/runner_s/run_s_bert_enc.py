import sys

import tensorflow as tf

from data_generator.NLI.enlidef import snli_train_size
from epath import job_man_dir
from job_manager.job_runner_with_server import JobRunnerS
from misc_lib import ceil_divide
from trainer_v2.arg_flags import flags_parser
from trainer_v2.epr.s_bert_enc import EncodeWorker, load_segmented_data


def main(arg):
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    with strategy.scope():
        dataset = "snli"
        split = "train"
        job_name = f"{dataset}_{split}_sbert_align"

        def jsonl_file_load_fn(job_id):
            return load_segmented_data(dataset, split, job_id)

        def factor(output_dir):
            return EncodeWorker(output_dir, arg.config_path, arg.init_checkpoint, jsonl_file_load_fn)

        num_jobs = ceil_divide(snli_train_size, 10000)
        job_runner = JobRunnerS(job_man_dir, num_jobs, job_name, factor, max_job_per_worker=4)
        job_runner.auto_runner()


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)

