from trainer.np_modules import *
from multiprocessing import Process, Queue
from trainer.queue_feader import QueueFeader
import threading
import pickle
from path import data_path
from data_generator.tokenizer_b import EncoderUnit
from data_generator.data_parser.trec import *
import random
import sys
from misc_lib import *
from data_generator.adhoc.ws import load_marco_queries
from adhoc.bm25 import get_bm25
from config.input_path import train_data_dir




class DataSampler:
    def __init__(self, queries, collection):
        self.collection = collection
        self.inv_index = get_inverted_index(self.collection)
        self.idf = Idf(list(self.collection.values()))
        self.threshold_boring_doc = 20
        self.min_posting = 5
        self.inst_per_query = 30
        self.queries = queries

    def save_to_pickle(self, pickle_name):
        save_to_pickle(self, pickle_name)

    @classmethod
    def init_from_pickle(cls, pickle_name):
        return load_from_pickle(pickle_name)

    def tfidf_span(self, q_terms, text_span):
        return sum([text_span.count(q_i) * self.idf[q_i] for q_i in q_terms])

    def check_worthy(self, q_terms, doc_id_list, seq_len):
        max_score = 0
        window_size = seq_len * 3
        for doc_id in doc_id_list:
            raw_document = self.collection[doc_id]
            loc_ptr = 0
            while loc_ptr < len(raw_document):
                text_span = raw_document[loc_ptr:loc_ptr + window_size]

                score = self.tfidf_span(q_terms, text_span)
                max_score = max(score, max_score)
                loc_ptr += window_size
        return max_score >= self.threshold_boring_doc

    def pair_generator(self, seq_len):
        ranked_list = self.ranked_list_generate(seq_len)
        for query, score_group in ranked_list:
            candidate = []
            for key_score, span_list in score_group.items():
                for score, span in span_list:
                    candidate.append((score, span))
            # print("candidate len : {}".format(len(candidate)))
            for i in range(self.inst_per_query):
                l = random.sample(range(len(candidate)), 2)
                x1 = candidate[l[0]]
                x2 = candidate[l[1]]
                # print(x1[0], x1[1][:100])
                # print(x2[0], x2[1][:100])
                if x1[0] < x2[0]:
                    yield query, x1, x2
                else:
                    yield query, x2, x1

    def ranked_list_generate(self, seq_len):
        def flatten_and_get_doc_id(postings_list):
            doc_ids = []
            for postings in postings_list:
                for doc_id, idx in postings:
                    doc_ids.append(doc_id)
            return doc_ids

        def sample_size():
            window_size = seq_len * 3
            return random.randrange(0, window_size * 10)

        def sample_shift():
            return sample_size()

        def sample_debiase(tf_n_span):
            max_occurence = 2

            output = {}
            for score, span in tf_n_span:
                score_grouper = int(score + 0.8)
                if score_grouper not in output:
                    output[score_grouper] = []
                if len(output[score_grouper]) < max_occurence:
                    output[score_grouper].append((score, span))
            return output

        collection_len = 252359881
        avdl = collection_len / len(self.collection)

        random.shuffle(self.queries)
        for query in self.queries:
            q_terms = query.split()
            postings_list = []
            for qterm in q_terms:
                postings = self.inv_index[qterm]
                if len(postings) < self.min_posting:
                    break  # Skip this query
                postings_list.append(postings)

            if not postings_list:
                continue
            print("Query :", query)
            doc_id_list = flatten_and_get_doc_id(postings_list)
            print("Docs : {}".format(len(doc_id_list)))

            spans = []
            if len(doc_id_list) > 1000:
                doc_id_list = random.sample(doc_id_list, 1000)

            if not self.check_worthy(q_terms, doc_id_list, seq_len):
                continue

            # Scan docs and retrieve spans
            for doc_id in doc_id_list:
                raw_document = self.collection[doc_id]
                loc_ptr = sample_shift()
                while loc_ptr < len(raw_document):
                    window_size = sample_shift()
                    text_span = raw_document[loc_ptr:loc_ptr + window_size]
                    score = get_bm25(" ".join(q_terms), text_span, self.idf.df, N=len(self.collection), avdl=avdl)
                    spans.append((score, text_span))
                    loc_ptr += sample_shift()
            score_group = sample_debiase(spans)
            for key in score_group:
                print(key)
            yield query, score_group


class DataWriter:
    def __init__(self, max_sequence):
        tprint("Loading data sampler")
        #mem_path = "/dev/shm/robust04.pickle"
        #self.data_sampler = pickle.load(open(mem_path, "rb"))
        self.data_sampler = DataSampler.init_from_pickle("samplerV")
        vocab_filename = "bert_voca.txt"
        voca_path = os.path.join(data_path, vocab_filename)
        self.encoder_unit = EncoderUnit(max_sequence, voca_path)
        self.pair_generator = self.data_sampler.pair_generator(max_sequence)


    def encode_pair(self, instance):
        query, case1, case2 = instance
        for y, sent in [case1, case2]:
            entry = self.encoder_unit.encode_pair(query, sent)
            yield entry["input_ids"], entry["input_mask"], entry["segment_ids"]

    def get_data(self, data_size):
        assert data_size % 2 == 0
        result = []
        ticker = TimeEstimator(data_size, sample_size=100)
        while len(result) < data_size:
            raw_inst = self.pair_generator.__next__()
            result += list(self.encode_pair(raw_inst))
            ticker.tick()
        return result

    def write(self, path, num_data):
        assert num_data % 2 == 0
        pickle.dump(self.get_data(num_data), open(path, "wb"))



def write_data():
    random.seed()
    #start_i = int(sys.argv[1])
    start_i = 0
    print("data:", start_i)
    seq_len = 2000
    block_len = 16 * 1000  # it will be about 200 MB
    dw = DataWriter(seq_len)

    filename = "data{}.pickle".format(start_i)
    path = os.path.join(data_path, "robust_train_4", filename)
    dw.write(path, block_len)


def init_sampler_robust04():
    marco_queries = list(load_marco_queries())
    robust_colleciton = load_robust(robust_path)
    data_sampler = DataSampler(marco_queries, robust_colleciton)
    save_to_pickle(data_sampler, "samplerV")

if __name__ == '__main__':
    write_data()
