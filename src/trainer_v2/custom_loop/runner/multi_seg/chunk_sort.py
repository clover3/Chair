import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from trainer_v2.custom_loop.neural_network_def.lattice_based import MonoSortCombiner


from trainer_v2.custom_loop.runner.nli_asymmetric.two_seg_commons import two_seg_common2

from trainer_v2.custom_loop.neural_network_def.ctx2 import CtxChunkInteraction
from trainer_v2.chair_logging import c_log


if __name__ == "__main__":
    c_log.info("Start {}".format(__file__))
    inner = CtxChunkInteraction(MonoSortCombiner)
    two_seg_common2(inner)