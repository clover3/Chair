import sys

from taskman_client.wrapper3 import report_run3
from trainer_v2.custom_loop.runner.nli_siamese_p.mean_pooling_mains_common import mean_pooling_common
from trainer_v2.train_util.arg_flags import flags_parser


@report_run3
def main(args):
    project_dim = 16 * 728
    mean_pooling_common(args, project_dim)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)


