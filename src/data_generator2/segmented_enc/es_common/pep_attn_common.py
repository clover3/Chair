from abc import ABC, abstractmethod
from typing import Tuple, OrderedDict

import numpy as np

from data_generator2.segmented_enc.es_common.es_two_seg_common import PairData
from trainer_v2.per_project.transparency.mmp.attn_compute.iter_attn import QDWithScoreAttn

PairWithAttn = Tuple[PairData, np.ndarray]


class PairWithAttnEncoderIF(ABC):
    @abstractmethod
    def encode_fn(self, e: Tuple[PairWithAttn, PairWithAttn]) -> OrderedDict:
        pass


class QDWithAttnEncoderIF(ABC):
    @abstractmethod
    def encode_fn(self, e: Tuple[QDWithScoreAttn, QDWithScoreAttn]) -> OrderedDict:
        pass