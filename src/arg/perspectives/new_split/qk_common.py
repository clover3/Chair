from typing import List, Iterable, Iterator

from arg.perspectives.load import splits
from arg.qck.decl import QKUnit
from cache import load_from_pickle
from list_lib import flatten


def load_all_qk() -> List[QKUnit]:
    itr_qk_candidate_names: Iterator[str] = map("pc_qk2_{}".format, splits)
    qk_unit_itr: Iterator[List[QKUnit]] = map(load_from_pickle, itr_qk_candidate_names)
    all_qk_units: Iterable[QKUnit] = flatten(qk_unit_itr)
    return list(all_qk_units)






