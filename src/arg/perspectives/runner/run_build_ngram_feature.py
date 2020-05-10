import os
import sys
from collections import Counter
from typing import List

from arg.perspectives.n_gram_feature_collector import PCNGramFeature, PCVectorFeature
from cache import load_from_pickle, load_pickle_from, save_to_pickle


def build_ngram_feature(dir_path, st, ed):
    selected_ngram_set = load_from_pickle("selected_ngram_feature")
    ngram_range = [1, 2, 3]
    all_data_point = []
    for i in range(st, ed):
        file_path = os.path.join(dir_path, str(i))
        features: List[PCNGramFeature] = load_pickle_from(file_path)

        for f in features:
            vector_builder = []
            for n in ngram_range:
                counter: Counter = f.n_grams[n]
                vector = [counter[k] for k in selected_ngram_set[n]]
                vector_builder.extend(vector)

            r = PCVectorFeature(f.claim_pers, vector_builder)
            all_data_point.append(r)

    save_name = os.path.basename(dir_path) + "_ngram_features"
    save_to_pickle(all_data_point, save_name)


if __name__ == "__main__":
    build_ngram_feature(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
