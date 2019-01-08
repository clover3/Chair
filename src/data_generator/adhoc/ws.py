from data_generator.tokenizer_b import FullTokenizerWarpper, _truncate_seq_pair
from data_generator.text_encoder import SubwordTextEncoder, TokenTextEncoder, CLS_ID, SEP_ID, EOS_ID
import xml.etree.ElementTree as ET

import tensorflow as tf
import csv
import math
from path import data_path
from collections import Counter, defaultdict
from cache import *
from evaluation import *
from data_generator.data_parser.trec import load_trec
num_classes = 3
import random

corpus_dir = os.path.join(data_path, "adhoc")
trecText_path = os.path.join(corpus_dir, "trecText")


def gen_trainable_iterator(n_per_query):
    doc_sampling = query_judge()

    for query, stats in doc_sampling:
        candidate = []
        for score_group, span_list in stats.items():
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

class EncoderUnit:
    def __init__(self, max_sequence, voca_path):
        self.encoder = FullTokenizerWarpper(voca_path)
        self.max_seq = max_sequence

    def encode_long_text(self, query, text):
        tokens_a = self.encoder.encode(query)
        tokens_b = self.encoder.encode(text)
        max_b_len = self.max_seq -(len(tokens_a) + 3)
        idx  = 0
        result = []
        while idx < len(tokens_b):
            sub_tokens_b = tokens_b[idx:idx+max_b_len]
            result.append(self.encode_inner(tokens_a, sub_tokens_b))
            idx += max_b_len
        return result

    def encode_inner(self, tokens_a, tokens_b):
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, self.max_seq - 3)

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append(CLS_ID)
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append(SEP_ID)
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append(SEP_ID)
            segment_ids.append(1)

        input_ids = tokens

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self.max_seq:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == self.max_seq
        assert len(input_mask) == self.max_seq
        assert len(segment_ids) == self.max_seq

        return {
            "input_ids": input_ids,
            "input_mask":input_mask,
            "segment_ids": segment_ids
        }

class DataLoader:
    def __init__(self, max_sequence, vocab_filename, voca_size):
        self.train_data = None
        self.dev_data = None
        self.test_data = None

        inst_per_query = 20
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


def load_trec_queries():
    query_path = os.path.join(corpus_dir, "test_query")

    buffer = open(query_path, "r").read()
    buffer = "<root>" + buffer + "</root>"
    root = ET.fromstring(buffer)

    for top in root:
        query_id = top[0].text
        query = top[1].text
        yield query_id, query


def load_queries():
    query_path = os.path.join(corpus_dir, "queries.train.tsv")
    f = open(query_path, "r")
    for row in csv.reader(f, delimiter="\t"):
        yield row[1]


def get_inverted_index(collection):
    result = defaultdict(list)
    for doc_id, doc in collection.items():
        for word_pos, word in enumerate(doc.split()):
            result[word].append((doc_id, word_pos))

    return result

def get_tf_index(inv_index):
    result = dict()
    for term, posting_list in inv_index.items():
        count_list = Counter()
        for doc_id, word_pos in posting_list:
            count_list[doc_id] += 1
        result[term] = count_list

    return result



class Idf:
    def __init__(self, docs):
        self.df = Counter()
        self.idf = dict()
        for doc in docs:
            term_count = Counter()
            for token in doc.split():
                term_count[token] = 1
            for elem, cnt in term_count.items():
                self.df[elem] += 1
        N = len(docs)

        for term, df in self.df.items():
            self.idf[term] = math.log(N/df)
        self.default_idf = math.log(N/1)

    def __getitem__(self, term):
        if term in self.idf:
            return self.idf[term]
        else:
            return self.default_idf

def query_judge():
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
        max_occurence = 5

        output = {}
        for score, span in tf_n_span:
            score_grouper = int(score + 0.8)
            if score_grouper not in output:
                output[score_grouper] = []
            if len(output[score_grouper]) < max_occurence:
                output[score_grouper].append((score, span))
        return output


    queries = load_queries()
    for query in queries:
        q_terms = query.split()
        postings_list= []
        for qterm in q_terms:
            postings = inv_index[qterm]
            if len(postings) < 5:
                break # Skip this query
            postings_list.append(postings)

        print("Debug) Query ", query)
        if not postings_list:
            continue

        doc_id_list = flatten_and_get_doc_id(postings_list)

        spans = []
        if len(doc_id_list) > 1000:
            doc_id_list = random.sample(doc_id_list, 1000)
        for doc_id in doc_id_list:
            raw_document = collection[doc_id]
            loc_ptr = sample_shift()
            while loc_ptr < len(raw_document):
                text_span = raw_document[loc_ptr:loc_ptr + window_size]

                score = tfidf_span(q_terms, text_span)
                spans.append((score, text_span))
                loc_ptr += sample_shift()
        stats = sample_debiase(spans)
        yield query, stats






if __name__ == '__main__':
    print("Generator")
    load_trec_queries()