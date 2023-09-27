import sys

from taskman_client.wrapper2 import report_run_named
import time

from taskman_client.wrapper3 import report_run3
from trainer_v2.train_util.arg_flags import flags_parser


@report_run3
def main(args):
    print("Sleeping")
    time.sleep(10)
    print("Done")
    return NotImplemented


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)