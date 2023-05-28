from cpath import output_path
from misc_lib import path_join, get_second, read_non_empty_lines_stripped
import numpy as np

from trainer_v2.per_project.transparency.mmp.term_effect_measure_single_q_term.compute_gains_resourced import compute_gains_with


def main():
    idx = 0
    term_path = path_join(output_path, "msmarco", "terms10K_stemmed.txt")
    terms = read_non_empty_lines_stripped(term_path)
    corpus_name = f"when_full_re_{idx}"
    gain_arr = compute_gains_with(corpus_name, terms)

    gain_summary = np.sum(gain_arr, axis=0)
    paired = list(zip(terms, gain_summary))
    paired.sort(key=get_second, reverse=True)
    for t, s in paired:
        print(t, s)



if __name__ == "__main__":
    main()