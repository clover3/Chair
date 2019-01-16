from data_generator.adhoc.data_sampler import DataWriter
import os
from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler
import xmlrpc.client

import random
import csv
from path import data_path
from data_generator.tokenizer_b import EncoderUnit
from concurrent.futures import ProcessPoolExecutor, wait, ThreadPoolExecutor

def load_galago_judgement(path):
# Sample Format : 475287 Q0 LA053190-0016_1274 1 15.07645119 galago
    q_group = dict()
    for line in open(path, "r"):
        q_id, _, doc_id, rank, score, _ = line.split()
        if q_id not in q_group:
            q_group[q_id] = list()
        q_group[q_id].append((doc_id, rank, score))

def load_doc_list(path):
    doc_ids = set()
    for line in open(path, "r"):
        doc_ids.add(line.strip())
    return doc_ids

def load_marco_query(path):
    f = open(path, "r")
    queries = dict()
    for row in csv.reader(f, delimiter="\t"):
        q_id = row[0]
        query = row[1]
        queries[q_id] = query
    return queries


def debiased_sampling(ranked_list):
    output = {}
    for doc_id, rank, score in ranked_list:
        score_grouper = int(score + 0.8)
        if score_grouper not in output:
            output[score_grouper] = (doc_id, score)
    return output


class DataSampler:
    def __init__(self, doc_ids, judgement_path, query):
        # load doc_lisself.doc_ids = load_doc_list(doc_id_path)t
        self.q_group = load_galago_judgement(judgement_path)
        # load query-judgement
        self.doc_ids = doc_ids
        self.query = query
        self.n_sample_ranked = 5
        self.n_sample_not_ranked = 3

    def sample(self):
        # How much?
        sampled = []
        for q_id in self.q_group:
            query = self.query[q_id]
            ranked_list = self.q_group[q_id]
            if len(ranked_list) < 20 :
                continue

            sample_space = debiased_sampling(ranked_list)
            # Sample 5 pairs from ranked list
            # Sample 3 pairs, where one is from ranked_list and one is from other than ranked list

            for i in range(self.n_sample_ranked):
                (doc_id_1, score_1), (doc_id_2, score_2) = random.sample(sample_space, 2)

                if score_1 < score_2:
                    sampled.append((query, doc_id_1, doc_id_2))
                else:
                    sampled.append((query, doc_id_2, doc_id_1))

            for i in range(self.n_sample_not_ranked):
                doc_id_1 = random.sample(self.doc_ids, 1)
                doc_id_2, _, _ = random.sample(ranked_list, 1)
                sampled.append((query, doc_id_1, doc_id_2))

        return sampled

class MasterEncoder:
    def __init__(self, query_path, doc_id_path):
        self.docs = {}
        self.query = load_marco_query(query_path)
        self.doc_ids = load_doc_list(doc_id_path)
        self.slave_addr = [

        ]

    def encoder_thread(self, slave_id):
        st = slave_id * 10000
        ed = st + 10000

        judgement_dir = NotImplemented
        judgement_path = os.path.join(judgement_dir, "{}_{}.list".format(st, ed))
        slave_proxy = xmlrpc.client.ServerProxy(self.slave_addr[slave_id])
        slave_fn = slave_proxy.encode

        sampled = DataSampler(self.doc_ids, judgement_path, self.query).sample()

        payload = []
        for query, doc_id_1, doc_id_2 in sampled:
            payload.append((query, self.docs[doc_id_1], self.docs[doc_id_2]))

        result = slave_fn(payload)
        return result

    def task(self):
        with ThreadPoolExecutor(max_workers=20) as executor:
            for i in range(0, 10):
                executor.submit(self.encoder_thread, i)
            executor.shutdown(True)


class DataEncoder:
    vocab_filename = "bert_voca.txt"
    voca_path = os.path.join(data_path, vocab_filename)
    max_sequence = 200
    encoder_unit = EncoderUnit(max_sequence, voca_path)
    def __init__(self, port):
        self.port = port

    def encode(self, payload):
        result = []
        with ProcessPoolExecutor(max_workers=8) as executor:
            future_entry_list = []
            for query, text1, text2 in payload:
                for sent in [text1, text2]:
                    future_entry_list.append(executor.submit(self.encoder_unit.encode_pair, query, sent))
            for future_entry in future_entry_list:
                entry = future_entry.result()
                result.append((entry["input_ids"], entry["input_mask"], entry["segment_ids"]))
        return result

    def start_server(self):
        class RequestHandler(SimpleXMLRPCRequestHandler):
            rpc_paths = ('/RPC2',)

        print("Preparing server")
        server = SimpleXMLRPCServer(("0.0.0.0", self.port),
                                    requestHandler=RequestHandler,
                                    allow_none=True,
                                    )
        server.register_introspection_functions()

        server.register_function(self.encode, 'encode')
        server.serve_forever()


def data_sampler_test():
    doc_id_path = NotImplemented
    marco_path = NotImplemented
    doc_ids = load_doc_list(doc_id_path)
    query = load_marco_query(marco_path)
    judgement_path = ""
    sampler = DataSampler(doc_ids, judgement_path, query)
    sampler.sample()

def start_slave():
    port = 18202
    DataEncoder

if __name__ == '__main__':
    data_sampler_test()