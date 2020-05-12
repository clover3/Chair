import csv
import os
import pickle
import random
import sys
import xmlrpc.client
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from xmlrpc.server import SimpleXMLRPCRequestHandler
from xmlrpc.server import SimpleXMLRPCServer

from cpath import data_path
from data_generator.data_parser.trec import load_trec
from data_generator.tokenizer_b import EncoderUnit
from galagos.parse import load_galago_ranked_list
from misc_lib import pick1


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
    output = dict() 
    for doc_id, rank, score in ranked_list:
        score_grouper = int(float(score) + 0.8)
        if score_grouper not in output:
            output[score_grouper] = list()
        if len(output[score_grouper]) < 2:
            output[score_grouper].append((doc_id, score))
    return list(output.values())


class DataSampler:
    def __init__(self, doc_ids, judgement_path, query):
        print("DataSample init")
        # load doc_lisself.doc_ids = load_doc_list(doc_id_path)t
        self.q_group = load_galago_ranked_list(judgement_path)

        # load query-judgement
        self.doc_ids = list(doc_ids)
        self.query = query
        self.n_sample_ranked = 5
        self.n_sample_not_ranked = 3

    def sample(self):
        print("sampling...")
        # How much?
        sampled = []
        for q_id in self.q_group:
            query = self.query[q_id]
            ranked_list = self.q_group[q_id]
            if len(ranked_list) < 20 :
                continue

            sample_space = []
            for span_list in debiased_sampling(ranked_list):
                for score, span in span_list:
                    sample_space.append((score, span))
            # Sample 5 pairs from ranked list
            # Sample 3 pairs, where one is from ranked_list and one is from other than ranked list
            print(sample_space)

            for i in range(self.n_sample_ranked):
                (doc_id_1, score_1), (doc_id_2, score_2) = random.sample(sample_space, 2)
                print((score_1, score_2))

                if score_1 < score_2:
                    sampled.append((query, doc_id_1, doc_id_2))
                else:
                    sampled.append((query, doc_id_2, doc_id_1))

            for i in range(self.n_sample_not_ranked):
                doc_id_1 = pick1(self.doc_ids)
                doc_id_2, _, _ = pick1(ranked_list)
                sampled.append((query, doc_id_1, doc_id_2))
        

        return sampled

class MasterEncoder:
    def __init__(self, query_path, doc_id_path):
        print("Master : Loading corpus")
        self.docs = load_trec("/mnt/nfs/work3/youngwookim/code/adhoc/robus/rob04.split.txt", 2)
        self.query = load_marco_query(query_path)
        self.doc_ids = load_doc_list(doc_id_path)
        self.slave_addr = [
            "http://compute-0-1:18202",
            "http://compute-0-2:18202",
            "http://compute-0-3:18202",
            "http://compute-0-4:18202",
            "http://compute-0-1:18202",
            "http://compute-0-2:18202",
            "http://compute-0-3:18202",
            "http://compute-0-4:18202",
            "http://compute-0-6:18202",
            "http://compute-0-11:18202",
        ]

    def encoder_thread(self, slave_id):
        print("{}] Start".format(slave_id))
        st = slave_id * 10000
        ed = st + 10000

        judgement_dir = "/mnt/nfs/work3/youngwookim/code/adhoc/robus/jobs" 
        judgement_path = os.path.join(judgement_dir, "{}.list".format(st, ed))
        slave_proxy = xmlrpc.client.ServerProxy(self.slave_addr[slave_id])
        slave_fn = slave_proxy.encode

        print("{}] Load sampled".format(slave_id))
        #sampled = DataSampler(self.doc_ids, judgement_path, self.query).sample()
        sample_filename = os.path.join("/mnt/nfs/work3/youngwookim/code/adhoc/robus/pair_samples", "sampled_{}.pickle".format(st))
        sampled = pickle.load(open(sample_filename, "rb"))

        payload = []
        for query, doc_id_1, doc_id_2 in sampled:
            payload.append((query, self.docs[doc_id_1], self.docs[doc_id_2]))

        print("{}] Sent".format(slave_id))
        result = slave_fn(payload)
        pickle.dump(result, open("payload_{}.pickle".format(slave_id), "wb"))
        print("{}] Done".format(slave_id))
        return result

    def task(self):
        ed = 10
        with ThreadPoolExecutor(max_workers=20) as executor:
            for i in range(4, 9):
                r = executor.submit(self.encoder_thread, i)
            for i in range(4, 9):
                r.result()


class DataEncoder:
    vocab_filename = "bert_voca.txt"
    voca_path = os.path.join(data_path, vocab_filename)
    max_sequence = 200
    encoder_unit = EncoderUnit(max_sequence, voca_path)
    def __init__(self, port):
        self.port = port

    def encode(self, payload):
        result = []
        print("Encode...")
        with ProcessPoolExecutor(max_workers=8) as executor:
            future_entry_list = []
            for query, text1, text2 in payload:
                for sent in [text1, text2]:
                    future_entry_list.append(executor.submit(self.encoder_unit.encode_pair, query, sent))
            for future_entry in future_entry_list:
                entry = future_entry.result()
                result.append((entry["input_ids"], entry["input_mask"], entry["segment_ids"]))
        print("Done...")
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

def sample_job(name):
    doc_id_path = "/mnt/nfs/work3/youngwookim/code/adhoc/robus/split_titles.txt"
    marco_path = "/mnt/nfs/work3/youngwookim/code/adhoc/robus/queries.train.tsv"
    doc_ids = load_doc_list(doc_id_path)
    query = load_marco_query(marco_path)
    judgement_path ="/mnt/nfs/work3/youngwookim/code/adhoc/robus/jobs/{}.list".format(name)
    sampler = DataSampler(doc_ids, judgement_path, query)
    ret = sampler.sample()
    pickle.dump(ret, open("sampled_{}.pickle".format(name), "wb"))



def start_slave():
    port = 18202
    DataEncoder(port).start_server()

def start_master():
    doc_id_path = "/mnt/nfs/work3/youngwookim/code/adhoc/robus/split_titles.txt"
    marco_path = "/mnt/nfs/work3/youngwookim/code/adhoc/robus/queries.train.tsv"
    master = MasterEncoder(marco_path, doc_id_path)
    master.task()

if __name__ == '__main__':
    if sys.argv[1] == "slave":
        start_slave()
    elif sys.argv[1] == "sample":
        sample_job(sys.argv[2])
    #data_sampler_test()
    else:
        start_master()
