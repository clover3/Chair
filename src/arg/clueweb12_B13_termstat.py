import os
from collections import Counter

from cache import save_to_pickle, load_from_pickle
from cpath import data_path
from data_generator.tokenizer_wo_tf import get_tokenizer


def load_clueweb12_B13_termstat():
    f = open(os.path.join(data_path, "clueweb12_B13_termstat.txt"), "r")
    tf = Counter()
    df = Counter()
    for line in f:
        tokens = line.split("\t")
        assert len(tokens) == 3
        word = tokens[0]
        tf[word] = int(tokens[1])
        df[word] = int(tokens[2])
    return tf, df


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


if __name__ == "__main__":
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
