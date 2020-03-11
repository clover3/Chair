import os
import pickle
import sys
from collections import Counter

from nltk.util import ngrams

from cpath import output_path
from data_generator.tokenizer_wo_tf import is_continuation
from misc_lib import TimeEstimator, left
from models.classic.stopword import is_stopword
from tlm.ukp.sydney_data import ukp_load_tokens_for_topic, dev_pretend_ukp_load_tokens_for_topic


def get_ngram_pickle_path(topic, n):
    return os.path.join(output_path, "{}_{}gram".format(topic, n))


def load_n_gram_from_pickle(topic, n):
    return pickle.load(open(get_ngram_pickle_path(topic, n), "rb"))


def is_single_char_n_gram(ngram_item):
    return all([len(term) == 1 for term in ngram_item])


DROP_ONE_CHAR = "drop_one_char"
DROP_STOPWORD_BEG_END = "drop_stopword_beg_end"
MERGE_SUBWORD = "merge_subword"
WHITE_LIST = "white_list"


def is_stop_word_begin_or_ending(ngram_item):
    return is_stopword(ngram_item[0]) or is_stopword(ngram_item[-1])


def merge_subword(segment):
    def combine(tokens):
        if len(tokens) == 1:
            return tokens[0]
        s = ""
        for j, t in enumerate(tokens):
            if j == 0:
                s += t
            else:
                assert t[:2] == "##"
                s += t[2:]
        return s

    r = []
    cur_term = []
    for token in segment:
        if is_continuation(token):
            cur_term.append(token)
        else:
            r.append(combine(cur_term))
            cur_term = [token]
    return r





def build_ngram_lm_from_tokens_list(doc, n) -> Counter:
    tf = Counter()
    for segment in doc:
        tf.update(ngrams(segment, n))
    return tf



def count_n_gram_grom_docs(docs, n, config, exclude_fn):
    count = Counter()
    tick = TimeEstimator(len(docs))

    top_k = 10000

    after_pruning = False
    for doc_idx, doc in enumerate(docs):
        if doc_idx % 10000 == 0:
            print(doc_idx)
        tick.tick()
        for segment in doc:
            if MERGE_SUBWORD in config:
                segment = merge_subword(segment)
            assert type(segment) == list
            for ngram_item in ngrams(segment, n):
                if after_pruning and ngram_item in selected_ngram:
                    continue
                elif exclude_fn(ngram_item):
                    pass
                else:
                    count[ngram_item] += 1

        if len(count) > 1000 * 1000 and not after_pruning:
            print("Performing pruning")
            tf_cnt = list(count.items())
            tf_cnt.sort(key=lambda x: x[1], reverse=True)
            selected_ngram = set(left(tf_cnt[:top_k]))
            after_pruning = True

    return count


def load_n_1_gram_set(topic, n):
    if n == 1 :
        return set()
    else:
        count = load_n_gram_from_pickle(topic, n-1)
        l = list(count.items())
        l.sort(key=lambda x:x[1], reverse=True)
        top_k = 10000
        for j in range(n-1):
            top_k *= 100
        print(l[0])
        return set(left(l)[:top_k])


def count_n_gram_from_topic(n):
    topic = "abortion"
    #token_dict = ukp_load_tokens_for_topic(topic)
    token_dict = dev_pretend_ukp_load_tokens_for_topic(topic)
    n_1_gram_set = load_n_1_gram_set(topic, n)
    print("Loaded {} n-1 gram".format(len(n_1_gram_set)))

    def exclude_fn(ngram_item):
        skip = False
        if DROP_ONE_CHAR in config and is_single_char_n_gram(ngram_item):
            skip = True
        elif DROP_STOPWORD_BEG_END in config and is_stop_word_begin_or_ending(ngram_item):
            skip = True
        elif WHITE_LIST in config:
            skip = True
            if ngram_item[:-1] in n_1_gram_set or ngram_item[1:] in n_1_gram_set:
                skip = False
        return skip

    config = {}
    config[DROP_ONE_CHAR] = 1
    config[DROP_STOPWORD_BEG_END] = 1
    config[MERGE_SUBWORD] = 1
    config[WHITE_LIST] = 1
    count = count_n_gram_grom_docs(token_dict.values(), n, config, exclude_fn)
    pickle.dump(count, open(get_ngram_pickle_path(topic, n), "wb"))


def count_sents():
    topic = "abortion"
    token_dict = ukp_load_tokens_for_topic(topic)
    docs = token_dict.values()
    n_sents = 0
    for doc_idx, doc in enumerate(docs):
        n_sents += len(doc)
    print(n_sents)


if __name__ == "__main__":
    count_n_gram_from_topic(int(sys.argv[1]))
