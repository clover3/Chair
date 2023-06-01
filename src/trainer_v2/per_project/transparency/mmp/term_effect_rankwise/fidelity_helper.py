from dataclasses import dataclass
from typing import Tuple, List

from scipy.stats import pearsonr

@dataclass
class TermEffectPerQuery:
    target_scores: List[float]
    base_scores: List[float]
    changes: List[Tuple[int, float]]

    @classmethod
    def from_json(cls, j_obj):
        target_scores, base_scores, changes = j_obj
        return TermEffectPerQuery(target_scores, base_scores, changes)

    def to_json(self):
        return [self.target_scores, self.base_scores, self.changes]


def pearson_r_wrap(scores1: List[float], scores2: List[float]) -> float:
    r, p = pearsonr(scores1, scores2)
    return r


def compare_fidelity(
        te: TermEffectPerQuery,
        fidelity_fn
):
    target_score = te.target_scores
    old_score = te.base_scores
    new_score = list(te.base_scores)

    for idx, new_val in te.changes:
        new_score[idx] = new_val

    old_fidelity = fidelity_fn(target_score, old_score)
    new_fidelity = fidelity_fn(target_score, new_score)
    return old_fidelity, new_fidelity
