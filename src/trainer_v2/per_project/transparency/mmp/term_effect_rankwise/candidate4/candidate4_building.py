

# @report_run3
from collections import defaultdict

from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.misc_common import save_tsv
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper2 import get_cand4_path_helper, \
    MMPGAlignPathHelper, get_cand2_1_path_helper
from typing import List, Iterable, Callable, Dict, Tuple, Set


def main():
    c_log.info(__file__)

    # Select 10K query terms that are frequent
    k = 10000
    config: MMPGAlignPathHelper = get_cand4_path_helper()
    freq_q_terms: List[str] = config.load_qterm_candidates()
    print(freq_q_terms[:10])

    config2_1 = get_cand2_1_path_helper()
    term_pairs = config2_1.load_candidate_pairs()
    cand2_1_d = defaultdict(list)
    for q_term, d_term in term_pairs:
        cand2_1_d[q_term].append(d_term)

    def iter_candidates_output():
        for q_term_ngram in freq_q_terms[:k]:
            tokens = q_term_ngram.split()

            assert len(tokens) == 2
            cand_per_q_term = set()
            for q_term_word in tokens:
                d_term_list = cand2_1_d[q_term_word]
                cand_per_q_term.update(d_term_list)

            for d_term in cand_per_q_term:
                yield q_term_ngram, d_term

    save_path = config.per_pair_candidates.candidate_pair_path
    save_tsv(iter_candidates_output(), save_path)


if __name__ == "__main__":
    main()