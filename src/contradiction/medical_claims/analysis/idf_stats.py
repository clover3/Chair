import math
from collections import Counter

from krovetzstemmer import Stemmer

from adhoc.clueweb12_B13_termstat import load_clueweb12_B13_termstat_stemmed, clue_cdf
from cache import load_cache, save_to_pickle
from contradiction.medical_claims.token_tagging.problem_loader import load_alamri_split
from tab_print import print_table


def enum_alamri_as_tf(do_stem=True):
    problems = load_alamri_split("dev")
    stemmer = Stemmer()

    unique_texts = set()
    for p in problems:
        unique_texts.add(p.text1)
        unique_texts.add(p.text2)


    for text in unique_texts:
        tokens = text.split()
        if do_stem:
            token = map(stemmer.stem, tokens)
        yield Counter(tokens)


def ecc_stats():
    ret = load_cache("ecc_stat")
    if ret is None:
        ret = _ecc_stats()
        save_to_pickle(ret, "ecc_stat")
    return ret


def _ecc_stats():
    l_tf = Counter()
    l_df = Counter()
    for counter in enum_alamri_as_tf():
        for token, cnt in counter.items():
            l_tf[token] += cnt
            l_df[token] += 1
    return l_df, l_tf


def idf_to_category(idf_v):
    if idf_v < 2.8:
        return "Stopword"
    elif idf_v < 6:
        return "Daily"
    elif idf_v < 9:
        return "Topical"
    elif idf_v < 17:
        return "Rare"
    else:
        return "OOV"


def main():
    g_tf, g_df = load_clueweb12_B13_termstat_stemmed()

    def idf(df):
        return math.log((clue_cdf - df + 0.5) / (df + 0.5))

    l_df, l_tf = ecc_stats()


    voca = list(l_df.keys())
    d_list = []
    for word in voca:
        d = {
            'word': word,
            'l_tf': l_tf[word],
            'l_df': l_df[word],
            'g_tf': g_tf[word],
            'g_idf': idf(g_df[word]),
            'group': idf_to_category(idf(g_df[word]))
        }
        d_list.append(d)

    columns = ["word", "l_tf", "l_df", "g_tf", "g_idf", "group"]
    d_list.sort(key=lambda x: x['g_idf'])
    table = []
    for i in range(0, len(d_list), 1):
        row = d_list[i]
        row = [i] + [row[key] for key in columns]
        table.append(row)

    print_table(table)


def main2():
    def idf(df):
        return math.log((clue_cdf - df + 0.5) / (df + 0.5))

    l_df, l_tf = ecc_stats()
    for counter in enum_alamri_as_tf():
        for token, cnt in counter.items():
            l_tf[token] += cnt
            l_df[token] += 1

    voca = list(l_df.keys())
    d_list = []
    for word in voca:
        d = {
            'word': word,
            'l_tf': l_tf[word],
            'l_df': l_df[word],
        }
        d_list.append(d)

    columns = ["word", "l_tf", "l_df",]
    d_list.sort(key=lambda x: x['l_tf'])
    table = []
    for i in range(0, len(d_list), 1):
        row = d_list[i]
        row = [i] + [row[key] for key in columns]
        table.append(row)

    print_table(table)



if __name__ == "__main__":
    main2()
