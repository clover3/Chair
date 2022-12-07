# INPUT : collection, ngrams
#
from collections import Counter

import math

from arg.claim_building.count_ngram import load_n_gram_from_pickle
from arg.claim_building.count_ngram_coocur import load_topic_something_from_pickle, CO_OCCUR
from adhoc.clueweb12_B13_termstat import load_subword_term_stat
from list_lib import left
from misc_lib import assign_default_if_not_exists
from models.classic.stopword import is_stopword


def get_co_occurence_inv_index(co_occurrence_info):
    inv_index = {}
    for key, cnt in co_occurrence_info.items():
        ngram1, ngram2 = key
        assign_default_if_not_exists(inv_index, ngram1, lambda: Counter())
        assign_default_if_not_exists(inv_index, ngram2, lambda: Counter())
        inv_index[ngram1][ngram2] += cnt
        inv_index[ngram2][ngram1] += cnt

    inv_index_sorted = {}

    for key, counter in inv_index.items():
        inv_index_sorted[key] = list(counter.most_common())
    return inv_index_sorted


class LastKMax:
    def __init__(self, k):
        self.k = k
        self.last_log = []

    def add(self, val):
        if len(self.last_log) == self.k:
            self.last_log = self.last_log[1:] + [val]
        else:
            self.last_log.append(val)

    def max(self):
        return max(self.last_log)


def load_top_ngram_sorted(topic, clueweb_idf):
    def influence(e):
        ngram, tf = e
        s = 0
        for term in ngram:
            s += tf * clueweb_idf[term]
        return s

    r = []
    top_k = 10000
    for n in range(1, 4):
        count = load_n_gram_from_pickle(topic, n)
        l = list(count.items())
        l.sort(key=influence, reverse=True)

        for j in range(100):
            ngram, tf = l[j]
            print("{}\t{}\t{}".format(ngram, tf, influence(l[j])))
        r.extend(l[:top_k])
    return left(r)


def df_to_idf(df):
    c_df = 2 * max(df.values())
    idf = Counter()
    for key, cnt in df.items():
        idf[key] = math.log(c_df/cnt)
    return idf

def process():
    topic = "abortion"
    print("load_subword_term_stat")
    clueweb_tf, clueweb_df = load_subword_term_stat()

    clueweb_idf = df_to_idf(clueweb_df)
    all_ngram = load_top_ngram_sorted(topic, clueweb_idf)
    # sort all ngram by sum of tf-idf. where tf is from the topic corpus, and idf is from global corpus
    # So this value represents the amount of influence of the term (or n-gram)

    covered_ngram = set()
    #co_occurrence_info = load_co_occur_from_pickle(topic)
    print("loading co_occurrence_info")
    co_occurrence_info = load_topic_something_from_pickle(topic, CO_OCCUR + str(2))
    inv_index_sorted = get_co_occurence_inv_index(co_occurrence_info)



    def select_starting_ngram():
        # all_ngram  is sorted
        for ngram in all_ngram:
            if ngram not in covered_ngram:
                return ngram
        raise IndexError("run out of n-gram")

    def get_occurrence(ngram1, ngram2):
        return co_occurrence_info[ngram1, ngram2] + co_occurrence_info[ngram2, ngram1]

    def has_non_stopword_overlap(ngram1, ngram2):
        intersection = set(ngram1).intersection(ngram2)
        for t in intersection:
            if not is_stopword(t):
                return True
        return False

    def enum_candidate(cur_ngram_set):
        seen = set()
        rank = 0
        max_rank = max([len(inv_index_sorted[ngram]) for ngram in cur_ngram_set])
        while rank < max_rank:
            for ngram in cur_ngram_set:
                if rank < len(inv_index_sorted[ngram]):
                    key, value = inv_index_sorted[ngram][rank]
                    if key not in seen:
                        yield key
                        seen.add(key)
            rank += 1

    def select_most_relevant_ngram(cur_ngram_set):
        best_ngram = None
        best_score = -1
        k = len(cur_ngram_set)
        last_k_max = LastKMax(k)
        for cand_ngram in enum_candidate(cur_ngram_set):
            score = 0
            skip = False
            for ngram in cur_ngram_set:
                if has_non_stopword_overlap(cand_ngram, ngram):
                    skip = True

                score_elem = get_occurrence(cand_ngram, ngram)
                last_k_max.add(score_elem)
                score += score_elem

            if not skip:
                print(score, cand_ngram)
                if score > best_score:
                    best_score = score
                    best_ngram = cand_ngram

            if best_score > last_k_max.max() * k :
                print("Now not promissing")
                print("Last k scores : ", last_k_max.last_log)
                print("best_score : ", best_score)
                break

        return best_ngram
    print("main loop")

    stop = False
    selected_ngram_list_list = []
    cur_ngram_list = [select_starting_ngram()]
    while not stop:
        new_ngram = select_most_relevant_ngram(cur_ngram_list)
        cur_ngram_list.append(new_ngram)

        if len(cur_ngram_list) >= 3:
            spec_sum = 0
            for ngram in cur_ngram_list:
                s = specificity(ngram, clueweb_tf)
                spec_sum += s
                print(ngram, s)
                covered_ngram.add(ngram)
            print(cur_ngram_list)
            print(spec_sum)
            selected_ngram_list_list.append(cur_ngram_list)
            cur_ngram_list = [select_starting_ngram()]

        if len(selected_ngram_list_list) > 10000:
            stop = True


def specificity(tokens, get_tf_prob):
    return sum([-math.log(get_tf_prob(t)) for t in tokens])


if __name__ == "__main__":
    process()
