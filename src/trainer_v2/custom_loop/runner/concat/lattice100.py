import sys

from taskman_client.wrapper3 import report_run3
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.runner.concat.run_concat_lattice import run_concat_lattice_common
from trainer_v2.train_util.arg_flags import flags_parser


@report_run3
def main(args):
    n_keypoint = 4
    c_log.info("Start {}".format(__file__))
    run_concat_lattice_common(args, n_keypoint)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)


