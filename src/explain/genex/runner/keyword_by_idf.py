import os
from collections import Counter
from typing import List, Iterable

import math

from cpath import output_path
from explain.genex.idf_lime import load_idf_fn_for
from explain.genex.load import load_as_tokens, QueryDoc
from list_lib import dict_key_map
from misc_lib import tprint, TimeEstimator
from tlm.retrieve_lm.stem import CacheStemmer


def get_idf_keyword_score(problems: List[QueryDoc], get_idf) -> Iterable[Counter]:
    stemmer = CacheStemmer()
    ticker = TimeEstimator(len(problems))
    for p in problems:
        tokens = p.doc
        tf = Counter()
        reverse_map = {}  # Stemmed -> raw
        tokens = [t for t in tokens if t not in [".", ",", "!"]]
        for raw_t in tokens:
            stem_t = stemmer.stem(raw_t)
            reverse_map[stem_t] = raw_t
            tf[stem_t] += 1

        score_d = Counter()
        for term, cnt in tf.items():

            score = math.log(1+cnt) * get_idf(term)
            assert type(score) == float
            score_d[term] = score

        score_d_surface_form: Counter = Counter(dict_key_map(lambda x: reverse_map[x], score_d))
        ticker.tick()
        yield score_d_surface_form


def save_score_to_file(scores_list: Iterable[Counter], save_path):
    out_f = open(save_path, 'w')
    for scores in scores_list:
        answer = " ".join([term for term, _ in scores.most_common(15)])
        out_f.write(answer + "\n")
    out_f.close()


def main():
    data_name = "wiki"
    tprint("Loading idf scores")
    get_idf = load_idf_fn_for(data_name)
    problems: List[QueryDoc] = load_as_tokens(data_name)
    save_name = "{}_idf.txt".format(data_name)
    save_path = os.path.join(output_path, "genex", save_name)
    scores_list: Iterable[Counter] = get_idf_keyword_score(problems, get_idf)
    save_score_to_file(scores_list, save_path)


if __name__ == "__main__":
    main()