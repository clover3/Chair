from typing import Tuple, List

from scipy.stats import pearsonr

TermEffectPerQuery = Tuple[List[float], List[float], List[Tuple[int, float]]]


def pearson_r_wrap(scores1: List[float], scores2: List[float]) -> float:
    r, p = pearsonr(scores1, scores2)
    return r


def compare_fidelity(
        term_effect_per_query: TermEffectPerQuery,
        fidelity_fn
):
    target_score, old_score, changes = term_effect_per_query
    new_score = list(old_score)

    for idx, new_val in changes:
        new_score[idx] = new_val

    old_fidelity = fidelity_fn(target_score, old_score)
    new_fidelity = fidelity_fn(target_score, new_score)
    return old_fidelity, new_fidelity