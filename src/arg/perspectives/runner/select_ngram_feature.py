import os
import sys
from collections import Counter
from typing import List

from arg.perspectives.n_gram_feature_collector import PCNGramFeature
from cache import load_pickle_from, save_to_pickle
from list_lib import left


def select_ngram_feature(dir_path, st, ed, save_name):
    ngram_range = [1, 2, 3]
    df_counter = collect_ngram_count(dir_path, ed, ngram_range, st)
    k_select = 100

    selected_ngram_set = {}
    for n in ngram_range:
        for word, cnt in df_counter[n].most_common(k_select):
            print(word, cnt)

        words = left(list(df_counter[n].most_common(k_select)))
        selected_ngram_set[n] = words

    save_to_pickle(selected_ngram_set, save_name)


def collect_ngram_count(dir_path, ed, ngram_range, st):
    all_counter = {}
    df_counter = {}
    for n in ngram_range:
        all_counter[n] = Counter()
        df_counter[n] = Counter()
    for i in range(st, ed):
        file_path = os.path.join(dir_path, str(i))
        features: List[PCNGramFeature] = load_pickle_from(file_path)

        for f in features:
            for n in ngram_range:
                counter: Counter = f.n_grams[n]
                all_counter[n].update(counter)
                for key in counter:
                    df_counter[n][key] += 1
    return df_counter


if __name__ == "__main__":
    select_ngram_feature(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])