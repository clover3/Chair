from cache import load_pickle_from
from list_lib import pairzip
from trainer_v2.per_project.transparency.mmp.term_effect_measure.path_helper import get_gain_save_path
from cpath import output_path
from misc_lib import path_join, read_non_empty_lines_stripped, get_second
import numpy as np


def get_galign_label():
    pass


def compute_gain_10K_when():
    run_name = "10K"
    term_path = path_join(output_path, "msmarco", "terms10K_stemmed.txt")
    terms = read_non_empty_lines_stripped(term_path)

    all_arr = []
    for i in range(16):
        corpus_name = f"when_full_re_{i}"
        arr = load_pickle_from(get_gain_save_path(corpus_name, run_name))
        all_arr.append(arr)

    arr = np.concatenate(all_arr, axis=0)
    gain = np.sum(arr, axis=0)
    term_gain = pairzip(terms, gain)
    return term_gain


def main():
    term_gain = compute_gain_10K_when()
    term_gain.sort(key=get_second, reverse=True)

    for i in range(0, len(term_gain), 10):
        print(i, term_gain[i])


    return NotImplemented


if __name__ == "__main__":
    main()