from typing import Dict, List

from arg.perspectives.eval_caches import get_candidate_pickle_name
from arg.perspectives.eval_helper import get_extended_eval_candidate
from arg.perspectives.evaluate import perspective_getter
from arg.perspectives.load import splits
from arg.qck.decl import QCKCandidate
from cache import save_to_pickle
from list_lib import lmap, dict_value_map, dict_key_map


def get_extended_eval_candidate_as_qck_raw(split) -> Dict[str, List[QCKCandidate]]:
    c: Dict[int, List[int]] = get_extended_eval_candidate(split)

    def convert_candidates(candidates: List[int]) -> List[QCKCandidate]:
        p_texts = lmap(perspective_getter, candidates)
        l: List[QCKCandidate] = []
        for pid, text in zip(candidates, p_texts):
            l.append(QCKCandidate(str(pid), text))
        return l

    c2: Dict[int, List[QCKCandidate]] = dict_value_map(convert_candidates, c)
    return dict_key_map(str, c2)


if __name__ == "__main__":
    for split in splits:
        c = get_extended_eval_candidate_as_qck_raw(split)
        save_to_pickle(c, get_candidate_pickle_name(split))
