from typing import List, Tuple

import numpy as np


def convert_align_to_mismatch(align_score) -> Tuple[List[float], List[float]]:
    seg1_mismatch = 1 - np.max(align_score, axis=1)
    seg2_mismatch = 1 - np.max(align_score, axis=0)
    return seg1_mismatch.tolist(), seg2_mismatch.tolist()
