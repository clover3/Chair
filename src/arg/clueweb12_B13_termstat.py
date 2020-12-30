import os
from collections import Counter, defaultdict

from cache import save_to_pickle, load_from_pickle
from cpath import data_path
from data_generator.tokenizer_wo_tf import get_tokenizer


# modified

def load_clueweb12_B13_termstat():
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



def load_clueweb12_B13_termstat_stemmed():
    from krovetzstemmer import Stemmer
    stemmer = Stemmer()
    tf, df = load_clueweb12_B13_termstat()
    new_tf = Counter()

    for key, cnt in tf.items():
        new_tf[stemmer.stem(key)] += cnt
        pass

    df_info = defaultdict(list)
    for key, cnt in df.items():
        df_info[stemmer.stem(key)].append(cnt)

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
