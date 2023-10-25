from abc import ABC, abstractmethod
from typing import Tuple, OrderedDict

import numpy as np

from data_generator2.segmented_enc.es_common.es_two_seg_common import PairData

PairWithAttn = Tuple[PairData, np.ndarray]


class PairWithAttnEncoderIF(ABC):
    @abstractmethod
    def encode_fn(self, e: Tuple[PairWithAttn, PairWithAttn]) -> OrderedDict:
        pass