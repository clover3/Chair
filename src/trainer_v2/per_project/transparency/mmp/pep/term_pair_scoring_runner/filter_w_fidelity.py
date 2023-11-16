import sys
from typing import List, Dict, Tuple

from omegaconf import OmegaConf

from misc_lib import group_by, get_first, get_second
from trainer_v2.per_project.transparency.misc_common import read_term_pair_table_w_score, save_term_pair_scores


def select_within_std(values):
    # Calculate mean and standard deviation
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    std_dev = variance ** 0.5

    # Select values within one standard deviation from the mean
    lower_bound = mean - std_dev
    upper_bound = mean + std_dev
    selected_values = [idx for idx, x in enumerate(values) if lower_bound <= x]
    return selected_values, mean, std_dev


def analyze(
        pep_scores: List[Tuple[str, str, float]],
        fidelity: List[Tuple[str, str, float]],
        self_term_score: List[Tuple[str, str, float]],
        save_path):
    self_score_d = {}
    for q_term, q_term_same, score in self_term_score:
        assert q_term == q_term_same
        self_score_d[q_term] = float(score)

    fidelity_grouped: Dict[str, List[Tuple]] = group_by(fidelity, get_first)
    grouped: Dict[str, List[Tuple]] = group_by(pep_scores, get_first)
    pep_score_d = {(qt, dt): s for qt, dt, s in pep_scores}
    new_hypo_gain_sum = 0
    out_entry = []

    for q_term, entries in fidelity_grouped.items():
        self_score: float = self_score_d[q_term]
        candidates = []
        for _q_term, d_term, fidelity_score in entries:
            candidates.append((d_term, pep_score_d[(q_term, d_term)], fidelity_score))

        candidates.sort(key=get_second, reverse=True)
        gain_acc = 0
        max_gain = 0
        pep_when_max = 999
        for d_term, pep, fidelity_score in candidates:

            gain_acc += fidelity_score
            if gain_acc > max_gain:
                max_gain = gain_acc
                pep_when_max = pep

        n_used = 0
        new_hypo_gain = 0
        for d_term, pep, fidelity_score in candidates:
            if pep >= pep_when_max or pep >= self_score:
                new_hypo_gain += fidelity_score
                out_entry.append((q_term, d_term, pep))
                n_used += 1

        print("{0} {1:.2f} {2:.2f} {3:.2f} {4} {5}".
              format(q_term, self_score, pep_when_max, max_gain, n_used, len(candidates)))
        new_hypo_gain_sum += new_hypo_gain
    # save_term_pair_scores(out_entry, save_path)


        # ys.append(mean)
        # xs.append(self_score)
    # plt.ylim(-1, 1)


def from_path(term_pair_score_path,
              fidelity_path,
              self_term_score_path,
              save_path
              ):
    term_pair_scores: List[Tuple[str, str, float]] = read_term_pair_table_w_score(term_pair_score_path)
    fidelity: List[Tuple[str, str, float]] = read_term_pair_table_w_score(fidelity_path)
    fidelity_non_zero = [(q, d, s) for q, d, s in fidelity if s < -0.1 or s > 0.1]
    self_term_score: List[Tuple[str, str, float]] = read_term_pair_table_w_score(self_term_score_path)

    analyze(term_pair_scores, fidelity_non_zero, self_term_score, save_path)


def main():
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)

    from_path(
        conf.term_pair_scores_path,
        conf.fidelity_path,
        conf.self_term_score_path,
        conf.save_path
    )


if __name__ == "__main__":
    main()
