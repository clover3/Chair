import pickle
import sys
from typing import List, Iterable, Callable, Dict, Tuple, Set

from omegaconf import OmegaConf

from cpath import output_path
from misc_lib import path_join, group_by, get_first

from misc_lib import get_second
from trainer_v2.per_project.transparency.misc_common import save_tsv, read_term_pair_table_w_score, save_term_pair_scores


def filter_high_scores_from_paths(term_pair_score_path, self_term_score_path, save_path):
    term_pair_scores: List[Tuple[str, str, float]] = read_term_pair_table_w_score(term_pair_score_path)
    self_term_score: List[Tuple[str, str, float]] = read_term_pair_table_w_score(self_term_score_path)
    filter_over_self_score(term_pair_scores, self_term_score, save_path)


def filter_over_self_score(
        term_pair_scores: List[Tuple[str, str, float]],
        self_term_score: List[Tuple[str, str, float]],
        save_path):
    self_score_d = {}
    for q_term, q_term_same, score in self_term_score:
        assert q_term == q_term_same
        self_score_d[q_term] = float(score)

    filter_over_threshold(term_pair_scores, self_score_d, save_path)


def filter_over_threshold(term_pair_scores, threshold_d, save_path):
    grouped: Dict[str, List[Tuple]] = group_by(term_pair_scores, get_first)
    out_entry = []
    for q_term, entries in grouped.items():
        entries: List[Tuple[str, str, float]] = entries
        self_score: float = threshold_d[q_term]
        for _q_term, d_term, score in entries:
            if score >= self_score and d_term != q_term:
                out_entry.append((q_term, d_term, score))
    save_term_pair_scores(out_entry, save_path)


def main():
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    filter_high_scores_from_paths(
        conf.term_pair_scores_path,
        conf.self_term_score_path,
        conf.save_path
    )


if __name__ == "__main__":
    main()
