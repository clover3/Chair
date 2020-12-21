from collections import Counter
from typing import List, Iterable, Dict, Tuple

from arg.qck.decl import QKUnit
from list_lib import lmap
from models.classic.lm_util import average_counters, tokens_to_freq


def kdp_to_lm(qk_units: List[QKUnit]) -> Dict[str, Counter]:
    def do(qk: QKUnit) -> Tuple[str, Counter]:
        q, kdp_list = qk
        tokens_itr: Iterable[List[str]] = [kdp.tokens for kdp in kdp_list]
        counter_list = lmap(tokens_to_freq, tokens_itr)
        counter = average_counters(counter_list)
        return q.query_id, counter

    return dict(map(do, qk_units))


