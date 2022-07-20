import os
import sys

from trainer_v2.custom_loop.runner.nli_asymmetric.two_seg_commons import two_seg_common

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from trainer_v2.custom_loop.neural_network_def.segmented_enc import FuzzyLogicLayer
from trainer_v2.custom_loop.neural_network_def.contextualized import BERTAsymmetricContextualizedSlice3
from taskman_client.wrapper3 import report_run3
from trainer_v2.chair_logging import c_log
from trainer_v2.train_util.arg_flags import flags_parser


@report_run3
def main(args):
    inner = BERTAsymmetricContextualizedSlice3(FuzzyLogicLayer)
    c_log.info("Start {}".format(__file__))
    two_seg_common(args, inner)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)