import sys

from taskman_client.wrapper3 import report_run3
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.neural_network_def.combine_mat import MatrixCombineTrainable0
from trainer_v2.custom_loop.runner.concat.concat_common import concat_common
from trainer_v2.train_util.arg_flags import flags_parser


@report_run3
def main(args):
    combiner = MatrixCombineTrainable0
    c_log.info("Start {}".format(__file__))
    return concat_common(args, combiner)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)


