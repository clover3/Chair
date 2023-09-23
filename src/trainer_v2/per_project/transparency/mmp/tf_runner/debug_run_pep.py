import os
from trainer_v2.chair_logging import c_log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys

from taskman_client.wrapper3 import report_run3
from trainer_v2.train_util.arg_flags import flags_parser


@report_run3
def main(args):
    print("hi3")
    c_log.info(__file__)
    c_log.info("Hi")


if __name__ == "__main__":
    print("hi")
    args = flags_parser.parse_args(sys.argv[1:])
    print("hi2")

    main(args)
