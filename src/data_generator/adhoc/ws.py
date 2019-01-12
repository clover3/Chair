from data_generator.tokenizer_b import FullTokenizerWarpper, _truncate_seq_pair, EncoderUnit
from data_generator.text_encoder import SubwordTextEncoder, TokenTextEncoder, CLS_ID, SEP_ID, EOS_ID

import tensorflow as tf
import csv
from path import data_path
from evaluation import *
num_classes = 3
import random
from data_generator.data_parser.trec import *

corpus_dir = os.path.join(data_path, "adhoc")


def gen_trainable_iterator(n_per_query):
    doc_sampling = sample_query_eval_tfidf()

    for query, score_group in doc_sampling:
        candidate = []
        for key_score, span_list in score_group.items():
            for score, span in span_list:
                candidate.append((score,span))
        for i in range(n_per_query):
            l = random.sample(range(len(candidate)), 2)
            x1 = candidate[l[0]]
            x2 = candidate[l[1]]
            if x1[0] < x2[0]:
                yield query, x1, x2
            else:
                yield query, x2, x1

class DataLoader:
    def __init__(self, max_sequence, vocab_filename, voca_size):
        self.train_data = None
        self.dev_data = None
        self.test_data = None

        inst_per_query = 30
        self.generator = gen_trainable_iterator(inst_per_query)
        self.iter = iter(self.generator)
        voca_path = os.path.join(data_path, vocab_filename)
        assert os.path.exists(voca_path)
        print(voca_path)

        self.lower_case = True
        self.sep_char = "#"
        self.encoder = FullTokenizerWarpper(voca_path)
        self.voca_size = voca_size
        self.dev_explain = None
        self.encoder_unit = EncoderUnit(max_sequence, voca_path)

    def get_train_data(self, data_size):
        assert data_size % 2 == 0
        result = []
        while len(result) < data_size:
            raw_inst = self.iter.__next__()
            result += list(self.encode_pair(raw_inst))

        return result

    def get_dev_data(self):
        result = []
        for i in range(160):
            raw_inst = self.iter.__next__()
            result += list(self.encode_pair(raw_inst))

        return result

    def encode_pair(self, instance):
        query, case1, case2 = instance

        for y, sent in [case1, case2]:
            entry = self.encode(query, sent)
            yield entry["input_ids"], entry["input_mask"], entry["segment_ids"], y

    def encode(self, query, text):
        tokens_a = self.encoder.encode(query)
        tokens_b = self.encoder.encode(text)
        return self.encoder_unit.encode_inner(tokens_a, tokens_b)



def load_marco_queries():
    query_path = os.path.join(corpus_dir, "queries.train.tsv")
    f = open(query_path, "r")
    for row in csv.reader(f, delimiter="\t"):
        yield row[1]


def sample_query_eval_tfidf():
    def flatten_and_get_doc_id(postings_list):
        doc_ids = []
        for postings in postings_list:
            for doc_id, idx in postings:
                doc_ids.append(doc_id)
        return doc_ids

    collection = load_trec(trecText_path)
    inv_index = get_inverted_index(collection)

    window_size = 200 * 3
    idf = Idf(collection.values())

    def sample_shift():
        return random.randrange(0, window_size * 4)

    def count_query_term(q_terms, text_span):
        return sum([text_span.count(q_i) for q_i in q_terms])

    def tfidf_span(q_terms, text_span):
        return sum([text_span.count(q_i) * idf[q_i] for q_i in q_terms])

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


    queries = load_marco_queries()
    for query in queries:
        q_terms = query.split()
        postings_list= []
        for qterm in q_terms:
            postings = inv_index[qterm]
            if len(postings) < 5:
                break # Skip this query
            postings_list.append(postings)

        if not postings_list:
            print("Skip Query :", query)
            continue
        print("Query :", query)
        doc_id_list = flatten_and_get_doc_id(postings_list)
        print("Docs : {}".format(len(doc_id_list)))

        spans = []
        if len(doc_id_list) > 1000:
            doc_id_list = random.sample(doc_id_list, 1000)

        max_score = 0
        for doc_id in doc_id_list:
            raw_document = collection[doc_id]
            loc_ptr = sample_shift()
            while loc_ptr < len(raw_document):
                text_span = raw_document[loc_ptr:loc_ptr + window_size]

                score = tfidf_span(q_terms, text_span)
                max_score = max(score, max_score)
                spans.append((score, text_span))
                loc_ptr += sample_shift()
        if max_score < 20:
            continue
        score_group = sample_debiase(spans)
        for score in score_group:
            print("{} : {}".format(score, len(score_group[score])))
        yield query, score_group

if __name__ == '__main__':
    print("Generator")
    load_mobile_queries()