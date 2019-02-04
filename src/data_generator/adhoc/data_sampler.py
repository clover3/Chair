from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler
import xmlrpc.client
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


    def pair_generator(self, seq_len):
        ranked_list = self.ranked_list_generate(seq_len)
        for query, score_group in ranked_list:
            candidate = []
            for key_score, span_list in score_group.items():
                for score, span in span_list:
                    candidate.append((score, span))
            #print("candidate len : {}".format(len(candidate)))
            for i in range(self.inst_per_query):
                l = random.sample(range(len(candidate)), 2)
                x1 = candidate[l[0]]
                x2 = candidate[l[1]]
                #print(x1[0], x1[1][:100])
                #print(x2[0], x2[1][:100])
                if x1[0] < x2[0]:
                    yield query, x1, x2
                else:
                    yield query, x2, x1
    
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

    def ranked_list_generate(self, seq_len):
        def flatten_and_get_doc_id(postings_list):
            doc_ids = []
            for postings in postings_list:
                for doc_id, idx in postings:
                    doc_ids.append(doc_id)
            return doc_ids

        window_size = seq_len * 3
        def sample_shift():
            return random.randrange(0, window_size * 4)

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
            postings_list= []
            for qterm in q_terms:
                postings = self.inv_index[qterm]
                if len(postings) < self.min_posting:
                    break # Skip this query
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
        self.data_sampler = DataSampler.init_from_pickle("robust04")
        vocab_filename = "bert_voca.txt"
        voca_path = os.path.join(data_path, vocab_filename)
        self.encoder_unit = EncoderUnit(max_sequence, voca_path)
        self.pair_generator = self.data_sampler.pair_generator()


    def encode_pair(self, instance):
        query, case1, case2 = instance
        for y, sent in [case1, case2]:
            entry =  self.encoder_unit.encode_pair(query, sent)
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

class Server:
    def __init__(self, port, max_sequence):
        self.port = port
        tprint("Loading data sampler")
        self.data_sampler = DataSampler.init_from_pickle("robust04")
        vocab_filename = "bert_voca.txt"
        voca_path = os.path.join(data_path, vocab_filename)
        self.encoder_unit = EncoderUnit(max_sequence, voca_path)
        self.pair_generator = self.data_sampler.pair_generator()


    def encode_pair(self, instance):
        query, case1, case2 = instance
        for y, sent in [case1, case2]:
            entry =  self.encoder_unit.encode_pair(query, sent)
            yield entry["input_ids"], entry["input_mask"], entry["segment_ids"], y

    def get_data(self, data_size):
        assert data_size % 2 == 0
        result = []
        while len(result) < data_size:
            raw_inst = self.pair_generator.__next__()
            result += list(self.encode_pair(raw_inst))
        return result


    def start(self):
        class RequestHandler(SimpleXMLRPCRequestHandler):
            rpc_paths = ('/RPC2',)

        tprint("Preparing server")
        server = SimpleXMLRPCServer(("0.0.0.0", self.port),
                                    requestHandler=RequestHandler,
                                    allow_none=True,
                                    )
        server.register_introspection_functions()

        server.register_function(self.get_data, 'get_data')
        tprint("Waiting")
        server.serve_forever()


def start_server(args):
    port, max_sequence = args
    server = Server(port, max_sequence)
    server.start()


def start_slaves(max_seq, port_start, n_slave):
    port_start = port_start
    proc_list = []
    for slave_id in range(n_slave):
        port = port_start + slave_id
        p = Process(target=start_server, args=((port, max_seq),))
        p.daemon = True
        p.start()
        proc_list.append(p)
    for p in proc_list:
        p.join()


class DataLoaderFromSocket:
    def __init__(self, port_start, batch_size, n_slave):
        self.port_start = port_start
        self.proxy = []
        self.queue_feeders = []
        self.batch_size = batch_size
        self.train_queue = Queue(maxsize=1000)
        addr = "http://sydney2.cs.umass.edu"
        for slave_id in range(n_slave):
            port = self.port_start + slave_id
            server_addr = '{}:{}'.format(addr, port)
            self.proxy.append(xmlrpc.client.ServerProxy(server_addr))
            t = threading.Thread(target=self.feed_queue, args=(slave_id,))
            t.daemon = True
            t.start()
            self.queue_feeders.append(t)

    def feed_queue(self, slave_id):
        print("feed_queue({})".format(slave_id))
        while True:
            batches = self.get_data(slave_id, self.batch_size, 10)
            for batch in batches:
                self.train_queue.put(batch, True)

    def get_dev_data(self, batch_size):
        n_batches = 10
        return self.get_data(0, batch_size, n_batches)

    def get_data(self, slave_id, batch_size, n_batches):
        result = self.proxy[slave_id].get_data(batch_size * n_batches)
        batches = get_batches_ex(result, batch_size, 3)
        return batches

    def get_train_batch(self):
        return self.train_queue.get(block=True)


class DataLoaderFromFile:
    def __init__(self, batch_size, voca_size):
        self.batch_size = batch_size
        self.voca_size = voca_size
        self.train_queue = Queue(maxsize=100)
        t = threading.Thread(target=self.feed_queue)
        t.daemon = True
        t.start()


        self.cur_idx = 0
        self.file_idx = 0
        self.cur_data = []
        self.load_next_data()

    def get_path(self, i):
        filename = "data{}.pickle".format(i)
        return os.path.join(train_data_dir, filename)



    def feed_queue(self):
        print("feed_queue()")
        while True:
            batches = self.get_data(self.batch_size, 10)
            for batch in batches:
                self.train_queue.put(batch, True)

    def get_dev_data(self):
        n_batches = 10
        return self.get_data(self.batch_size, n_batches)

    def load_next_data(self):
        path = self.get_path(self.file_idx)
        self.file_idx += 1
        next_path = self.get_path(self.file_idx)
        if not os.path.exists(next_path):
            print("WARNING next file is unavailable : {}".format(next_path))
        self.cur_data = pickle.load(open(path, "rb"))
        print("Loaded data {}".format(self.file_idx-1))
        self.cur_idx = 0
        return self.cur_data

    def get_data(self, batch_size, n_batches):
        st = self.cur_idx
        ed = self.cur_idx + batch_size * n_batches
        if ed > len(self.cur_data):
            self.cur_data = self.load_next_data()
            st = self.cur_idx
            ed = self.cur_idx + batch_size * n_batches
        result = self.cur_data[st:ed]
        batches = get_batches_ex(result, batch_size, 3)
        self.cur_idx = ed
        return batches

    def get_train_batch(self):
        return self.train_queue.get(block=True)


def write_data():
    random.seed()
    start_i = int(sys.argv[1])
    print("data:", start_i)
    seq_len = 2000
    block_len = 16 * 1000  # it will be about 20 MB
    dw = DataWriter(seq_len)

    filename = "data{}.pickle".format(start_i)
    path = os.path.join(data_path, "robust_train_4", filename)
    dw.write(path, block_len)


def init_sampler_robust04():
    marco_queries = list(load_marco_queries())
    robust_colleciton = load_robust(robust_path)
    data_sampler = DataSampler(marco_queries, robust_colleciton)
    save_to_pickle(data_sampler, "robust04")


if __name__ == '__main__':
    write_data()
