import math
from dataclasses import dataclass
from typing import Tuple, List

from krovetzstemmer import Stemmer
from scipy.stats import pearsonr, spearmanr

from misc_lib import path_join
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import get_te_save_name, \
    get_fidelity_save_name
from trainer_v2.per_project.transparency.misc_common import load_list_from_gz_jsonl, save_number_to_file


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
    if len(scores1) == 1 and len(scores2) == 1:
        return 0
    r, p = pearsonr(scores1, scores2)
    if math.isnan(r):
        r = 0
    return r


def spearman_r_wrap(scores1: List[float], scores2: List[float]) -> float:
    if len(scores1) == 1 and len(scores2) == 1:
        return 0

    r, p = spearmanr(scores1, scores2)
    if math.isnan(r):
        r = 0
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


def compute_save_fidelity_from_te(fidelity_fn, fidelity_save_dir, partition_list, qd_itr, te_save_dir):
    stemmer = Stemmer()
    for q_term, d_term in qd_itr:
        f_change_sum = 0
        for partition_no in partition_list:
            q_term_stm = stemmer.stem(q_term)
            d_term_stm = stemmer.stem(d_term)
            save_name = get_te_save_name(q_term_stm, d_term_stm, partition_no)
            save_path = path_join(te_save_dir, save_name)
            print(save_path)
            te_list: List[TermEffectPerQuery] = load_list_from_gz_jsonl(save_path, TermEffectPerQuery.from_json)
            f_change = compute_fidelity_change(fidelity_fn, te_list)

            f_change_sum += f_change

        save_name = get_fidelity_save_name(q_term, d_term)
        fidelity_save_path = path_join(fidelity_save_dir, save_name)
        save_number_to_file(fidelity_save_path, f_change_sum)



def main():
    scores1 = [0, 1, 2, 3, 4]
    scores2a = [0, 0.1, 0.2, 0.3, 0.4]
    scores2b = [0, 0.1, 0.2, 0.9, 0.99]

    for scores2 in [scores2a, scores2b]:
        r, p = pearsonr(scores1, scores2)
        print('pearson', r)
        r, p = spearmanr(scores1, scores2)
        print('spearman', r)


if __name__ == "__main__":
    main()


def compute_fidelity_change_pearson(te_list: List[TermEffectPerQuery]):
    fidelity_fn = pearson_r_wrap
    return compute_fidelity_change(fidelity_fn, te_list)


def compute_fidelity_change_spearman(te_list: List[TermEffectPerQuery]):
    fidelity_fn = spearman_r_wrap
    return compute_fidelity_change(fidelity_fn, te_list)


def compute_fidelity_change(fidelity_fn, te_list):
    fidelity_pair_list: List[Tuple[float, float]] = [compare_fidelity(te, fidelity_fn) for te in te_list]
    delta_sum = 0
    for t1, t2 in fidelity_pair_list:
        delta = t2 - t1
        delta_sum += delta
    return delta_sum