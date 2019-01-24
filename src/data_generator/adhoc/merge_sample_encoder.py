import os
from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler
import xmlrpc.client
from data_generator.data_parser.trec import load_trec
import random
import csv
from path import data_path
from data_generator.tokenizer_b import EncoderUnit
from concurrent.futures import ProcessPoolExecutor, wait, ThreadPoolExecutor
from trainer.promise import PromiseKeeper, MyPromise, list_future
import pickle
import sys


def load_marco_query(path):
    f = open(path, "r")
    queries = dict()
    for row in csv.reader(f, delimiter="\t"):
        q_id = row[0]
        query = row[1]
        queries[q_id] = query
    return queries

class MasterEncoder:
    def __init__(self, query_path, doc_id_path):
        print("Master : Loading corpus")
        self.docs = load_trec("/mnt/nfs/work3/youngwookim/code/adhoc/robus/rob04.split.txt", 2)
        self.query = load_marco_query(query_path)
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

    def split_text(self, content):
        idx = 0
        window_size = 200 * 3
        while idx < len(content):
            span = content[idx:idx + window_size]
            yield span

    def encoder_thread(self, slave_id):
        st = slave_id * 10000

        slave_proxy = xmlrpc.client.ServerProxy(self.slave_addr[slave_id])
        slave_fn = slave_proxy.encode

        print("{}] Load sampled".format(slave_id))
        dir_path = "/mnt/nfs/work3/youngwookim/code/adhoc/robus/pair_samples"
        sample_filename = os.path.join(dir_path, "merger_train_{}.pickle".format(slave_id))
        sampled = pickle.load(open(sample_filename, "rb"))

        pk = PromiseKeeper(slave_fn)
        X_list = list(range(10))
        y_list = []
        for x in X_list:
            y = MyPromise(x, pk).future()
            y_list.append(y)

        future_list = []
        for q_id_1, doc_id_1, q_id_2, doc_id_2 in sampled:
            q1 = self.query[q_id_1]
            runs_1 = []
            for span in self.split_text(self.docs[doc_id_1]):
                runs_1.append(MyPromise((q1, span), pk).future())

            q2 = self.query[q_id_2]
            runs_2 = []
            for span in self.split_text(self.docs[doc_id_2]):
                runs_2.append(MyPromise((q2, span), pk).future())

            future_list.append((runs_1, runs_2))

        payload = []
        pk.do_duty()
        for r1, r2 in future_list:
            payload.append( (list_future(r1), list_future(r2)) )

        print("{}] Sent".format(slave_id))
        pickle.dump(payload, open("doc_level_compare_{}.pickle".format(slave_id), "wb"))
        print("{}] Done".format(slave_id))
        return payload

    def task(self):
        ed = 10
        with ThreadPoolExecutor(max_workers=20) as executor:
            for i in range(ed):
                r = executor.submit(self.encoder_thread, i)
            for i in range(ed):
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
            for query, text, in payload:
                future_entry_list.append(executor.submit(self.encoder_unit.encode_pair, query, text))
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
    else:
        start_master()
