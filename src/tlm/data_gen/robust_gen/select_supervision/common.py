from typing import Dict, Tuple

from cache import load_from_pickle


def load_score_set1() -> Dict[Tuple[str, str, int], float]:
    prefix = "robust_2A_"
    d = {}
    for name in [301, 351, 401, 601]:
        save_name = prefix + str(name)
        d.update(load_from_pickle(save_name))
        save_name = prefix + str(name) + "_train"
        d.update(load_from_pickle(save_name))
    return d