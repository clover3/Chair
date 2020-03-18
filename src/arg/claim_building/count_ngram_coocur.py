import itertools
import os
import pickle
from collections import Counter

from nltk import ngrams

from arg.claim_building.count_ngram import load_n_gram_from_pickle
from cpath import output_path
from data_generator.job_runner import JobRunner, sydney_working_dir
from list_lib import left
from misc_lib import TimeEstimator
from tlm.ukp.sydney_data import ukp_load_tokens_for_topic

CO_OCCUR = "co_occur"
OCCURRENCE_PART = "occurence_part"

def get_topic_something_pickle_path(topic, some_name):
    return os.path.join(output_path, "{}_{}".format(topic, some_name))


def load_topic_something_from_pickle(topic, some_name):
    return pickle.load(open(get_topic_something_pickle_path(topic, some_name), "rb"))


def get_co_occur_part_pickle_path(topic, idx):
    name = CO_OCCUR + str(idx)
    return get_topic_something_pickle_path(topic, name)


def load_co_occur_from_pickle(topic):
    return pickle.load(open(get_topic_something_pickle_path(topic, CO_OCCUR), "rb"))


def get_ordered_pair_comb(n):
    for i in range(n):
        for j in range(i+1, n):
            yield i, j

WINDOW_SIZE = 'window_size'


def count_co_occurrence(docs, docs_len, n_gram_range, target_n_grams, config):
    tick = TimeEstimator(docs_len)
    window_size = config[WINDOW_SIZE]
    counter = Counter()
    all_occurrence = []
    for doc_idx, doc in enumerate(docs):
        if doc_idx % 10000 == 0:
            print(doc_idx)
        tick.tick()

        occurrence = {}
        max_doc_len = 1000
        doc_len = min(len(doc), max_doc_len)

        def enum_doc(doc):
            for idx,b in enumerate(doc):
                if idx >= max_doc_len:
                    break
                yield b

        for seg_idx, segment in enumerate(enum_doc(doc)):
            occurrence[seg_idx] = list()
            for n in n_gram_range:
                assert type(segment) == list
                for ngram_item in ngrams(segment, n):
                    if ngram_item in target_n_grams:
                        occurrence[seg_idx].append(ngram_item)
            if seg_idx > 1000:
                break

        for seg_idx, _ in enumerate(enum_doc(doc)):
            def get_ngram_in_cur_window():
                st = seg_idx - window_size
                ed = seg_idx + window_size
                r = []
                for j in range(st, ed+1):
                    if j >=0 and j < doc_len:
                        r.extend(occurrence[j])
                return r

            cur_ngrams = get_ngram_in_cur_window()
            if len(cur_ngrams) > 200:
                cur_ngrams = cur_ngrams[:200]

            for i, j in get_ordered_pair_comb(len(cur_ngrams)):
                key = cur_ngrams[i], cur_ngrams[j]
                counter[key] += 1

        all_occurrence.append(occurrence)

    return counter, all_occurrence


def get_top_ngrams_set(topic, top_k):
    r = []
    for n in range(1, 4):
        count = load_n_gram_from_pickle(topic, n)
        l = list(count.items())
        l.sort(key=lambda x: x[1], reverse=True)
        r.extend(l[:top_k])
    return set(left(r))


class Worker:
    def __init__(self, topic, dummy_output_path):
        self.token_dict = ukp_load_tokens_for_topic(topic)

    def work(self, job_id):
        n_gram_range = range(1, 4)
        config = {}
        config[WINDOW_SIZE] = 1
        interval = 1000
        st = job_id * interval
        ed = (job_id+1) * interval
        docs = itertools.islice(self.token_dict.values(), st, ed)
        docs_len = min(interval, len(self.token_dict))
        target_n_grams = get_top_ngrams_set(topic, 10000)
        counter, all_occurrence = count_co_occurrence(docs, docs_len, n_gram_range, target_n_grams, config)
        pickle.dump(counter, open(get_co_occur_part_pickle_path(topic, job_id), "wb"))
        name = OCCURRENCE_PART + str(job_id)
        pickle.dump(all_occurrence, open(get_topic_something_pickle_path(topic, name), "wb"))


if __name__ == "__main__":
    topic = "abortion"
    runner = JobRunner(sydney_working_dir, 160, "count_ngram_cocour_abortion", lambda x: Worker(topic, x))
    runner.start()


