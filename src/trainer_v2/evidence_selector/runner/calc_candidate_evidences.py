import os
import sys

from epath import job_man_dir
from job_manager.job_runner_with_server import JobRunnerS
from trainer_v2.custom_loop.run_config2 import get_run_config_for_predict
from trainer_v2.evidence_selector.calc_candidate_evidences import Worker
from trainer_v2.train_util.arg_flags import flags_parser


def main(args):
    # Load model
    run_config = get_run_config_for_predict(args)
    src_input_dir = os.path.join(job_man_dir, "evidence_candidate_gen")

    def factory(output_dir):
        return Worker(run_config, src_input_dir, output_dir)

    n_jobs = 40

    runner = JobRunnerS(job_man_dir, n_jobs, "evidence_candidate_calc", factory)
    runner.auto_runner()


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
