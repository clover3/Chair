import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from trainer_v2.custom_loop.neural_network_def.multi_segments import ChunkStartEncoder
from trainer_v2.custom_loop.runner.nli_asymmetric.two_seg_commons import two_seg_common2
from trainer_v2.custom_loop.neural_network_def.segmented_enc import FuzzyLogicLayerM2
from trainer_v2.chair_logging import c_log


if __name__ == "__main__":
    c_log.info("Start {}".format(__file__))
    inner = ChunkStartEncoder(FuzzyLogicLayerM2)
    two_seg_common2(inner)
