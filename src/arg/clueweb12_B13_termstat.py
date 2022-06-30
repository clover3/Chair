import math
import os
import pickle
from collections import Counter, defaultdict
from typing import Dict, Tuple

from krovetzstemmer import Stemmer

from cache import save_to_pickle, load_from_pickle, load_pickle_from, function_cache_wrap
from cpath import data_path
from data_generator.tokenizer_wo_tf import get_tokenizer


# modified

def load_clueweb12_B13_termstat() -> Tuple[Counter, Counter]:
    f = open(os.path.join(data_path, "clueweb12_B13_termstat.txt"), "r", encoding="utf-8")
    tf = Counter()
    df = Counter()
    for line in f:
        tokens = line.split("\t")
        assert len(tokens) == 3
        word = tokens[0]
        tf[word] = int(tokens[1])
        df[word] = int(tokens[2])
    return tf, df


cdf = 50 * 1000 * 1000
clue_cdf = cdf
clueweb12aa_collection_length = 25601339619


def load_clueweb12_B13_termstat_stemmed():
    fn = function_cache_wrap(_load_clueweb12_B13_termstat_stemmed, "clueweb12_B13_termstat_stemmed")
    return fn()


def _load_clueweb12_B13_termstat_stemmed() -> Tuple[Dict, Dict]:
    from krovetzstemmer import Stemmer
    stemmer = Stemmer()
    tf, df = load_clueweb12_B13_termstat()
    new_tf = Counter()

    n_error = 0
    for key, cnt in tf.items():
        try:
            new_tf[stemmer.stem(key)] += cnt
        except UnicodeDecodeError:
            n_error += 1

    if n_error / len(tf) > 0.1:
        print("{} of {} are error".format(n_error, len(tf)))

    n_error = 0
    df_info = defaultdict(list)
    for key, cnt in df.items():
        try:
            df_info[stemmer.stem(key)].append(cnt)
        except UnicodeDecodeError:
            pass

    new_df = Counter()
    for key, cnt_list in df_info.items():
        cnt_list.sort(reverse=True)
        discount = 1
        discount_factor = 0.3
        df_est = 0
        for cnt in cnt_list:
            df_est += cnt * discount
            discount *= discount_factor

        new_df[key] = int(df_est)
    return new_tf, new_df


def save_clueweb12_B13_termstat_stemmed():
    obj = load_clueweb12_B13_termstat_stemmed()
    pickle.dump(obj, open(os.path.join(data_path, "clueweb12_B13_termstat_stemmed.txt"), "wb"))


def load_clueweb12_B13_termstat_stemmed_from_pickle():
    return load_pickle_from(os.path.join(data_path, "clueweb12_B13_termstat_stemmed.pickle"))


def translate_word_tf_to_subword_tf(word_tf):
    tokenizer = get_tokenizer()

    out = Counter()
    for word in word_tf:
        sub_words = tokenizer.tokenize(word)
        for sw in sub_words:
            out[sw] += word_tf[word]
    return out


TERMSTAT_SUBWORD = "load_clueweb12_B13_termstat_subword"


def load_subword_term_stat():
    return load_from_pickle(TERMSTAT_SUBWORD)


def save_subword_termstat():
    tf, df = load_clueweb12_B13_termstat()
    print("tf[hi]", tf['hi'])
    print("df[hi]", df['hi'])

    print("max df : ", max(df.values()))

    tf = translate_word_tf_to_subword_tf(tf)
    df = translate_word_tf_to_subword_tf(df)

    save_to_pickle((tf,df), TERMSTAT_SUBWORD)
    print("Subword:")
    print("tf[hi]", tf['hi'])
    print("df[hi]", df['hi'])


if __name__ == "__main__":
    save_clueweb12_B13_termstat_stemmed()


class ClueIDF:
    def __init__(self):
        tf, df = load_clueweb12_B13_termstat_stemmed()
        self.df = df
        self.stemmer = Stemmer()
        self.cdf = clue_cdf

    def get_weight(self, token) -> float:
        stemmed_token = self.stemmer(token)
        df = self.df[stemmed_token]
        if df == 0:
            df = 10

        assert self.cdf - df + 0.5 > 0
        return math.log((self.cdf - df + 0.5) / (df + 0.5))