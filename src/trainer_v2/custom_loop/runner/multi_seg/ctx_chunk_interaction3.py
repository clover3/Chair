import os

from trainer_v2.custom_loop.neural_network_def.ctx2_vars import CtxChunkInteraction3

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from trainer_v2.custom_loop.runner.nli_asymmetric.two_seg_commons import two_seg_common2

from trainer_v2.custom_loop.neural_network_def.segmented_enc import FuzzyLogicLayerM
from trainer_v2.chair_logging import c_log


if __name__ == "__main__":
    c_log.info("Start {}".format(__file__))
    inner = CtxChunkInteraction3(FuzzyLogicLayerM)
    two_seg_common2(inner)