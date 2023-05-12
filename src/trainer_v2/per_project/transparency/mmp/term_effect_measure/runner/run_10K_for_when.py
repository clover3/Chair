import sys

from cpath import output_path
from misc_lib import path_join, get_second, read_non_empty_lines_stripped
import numpy as np

from trainer_v2.per_project.transparency.mmp.term_effect_measure.compute_gains_resourced import compute_gains_with
from trainer_v2.per_project.transparency.mmp.term_effect_measure.path_helper import save_gain


def main():
    idx = sys.argv[1]
    term_path = path_join(output_path, "msmarco", "terms10K_stemmed.txt")
    terms = read_non_empty_lines_stripped(term_path)
    run_name = "10K"
    corpus_name = f"when_full_re_{idx}"
    gain_arr = compute_gains_with(corpus_name, terms)
    save_gain(gain_arr, corpus_name, run_name)



if __name__ == "__main__":
    main()