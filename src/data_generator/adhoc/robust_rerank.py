from data_generator.adhoc.data_sampler import DataSampler, DataWriter
from data_generator.data_parser.robust import *

from concurrent.futures import ProcessPoolExecutor, wait
from data_generator.data_parser import trec
from data_generator.tokenizer_b import EncoderUnit
from path import data_path
from collections import defaultdict
from misc_lib import *
import pickle

def encode_pred_set(top_k):
    vocab_filename = "bert_voca.txt"
    max_sequence = 200
    voca_path = os.path.join(data_path, vocab_filename)
    encoder_unit = EncoderUnit(max_sequence, voca_path)
    collection = trec.load_robust(trec.robust_path)
    print("Collection has #docs :", len(collection))
    queries = load_robust04_query()
    ranked_lists = load_2k_rank()
    window_size = 200 * 3

    payload = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        future_list = []
        for q_id in ranked_lists:
            ranked = ranked_lists[q_id]
            ranked.sort(key=lambda x:x[1])
            assert ranked[0][1] == 1
            print(q_id)
            doc_ids = []
            for doc_id, rank, score, in ranked[:top_k]:
                doc_ids.append(doc_id)

                raw_query = queries[q_id]
                content = collection[doc_id]
                idx = 0
                f_list = []
                while idx < len(content):
                    span = content[idx:idx + window_size]
                    f = executor.submit(encoder_unit.encode_pair, raw_query, span)
                    idx += window_size
                    f_list.append(f)
                #runs_future = executor.submit(encoder_unit.encode_long_text, raw_query, content)
                future_list.append((doc_id, q_id, f_list))

        def get_flist(f_list):
            r = []
            for f in f_list:
                r.append(f.result())
            return r

        for doc_id, q_id, f_list in future_list:
            payload.append((doc_id, q_id, get_flist(f_list)))

    pickle.dump(payload, open("payload_B_{}.pickle".format(top_k), "wb"))


def write_payload_in_plain(top_k):
    vocab_filename = "bert_voca.txt"
    max_sequence = 200
    voca_path = os.path.join(data_path, vocab_filename)
    collection = trec.load_robust(trec.robust_path)
    print("Collection has #docs :", len(collection))
    queries = load_robust04_query()
    ranked_lists = load_2k_rank()
    window_size = 200 * 3

    payload = []
    for q_id in ranked_lists:
        ranked = ranked_lists[q_id]
        ranked.sort(key=lambda x: x[1])
        assert ranked[0][1] == 1
        doc_ids = []
        for doc_id, rank, score, in ranked[:top_k]:
            doc_ids.append(doc_id)

            raw_query = queries[q_id]
            content = collection[doc_id]
            idx = 0
            doc_query_list = []
            while idx < len(content):
                span = content[idx:idx + window_size]
                doc_query_list.append((raw_query, span))
                idx += window_size
            #runs_future = executor.submit(encoder_unit.encode_long_text, raw_query, content)
            payload.append((q_id, doc_id, doc_query_list))

    total_n = len(payload)


    step = 1000
    print(total_n)
    idx = 0
    while idx < total_n:
        name = "payload200_part_{}".format(idx)
        path = os.path.join(data_path, "robust", "payload_plain_parts", name)
        pickle.dump(payload[idx: idx+step], open(path, "wb") )
        idx += step


def check_pred_set():
    path = os.path.join(data_path, "robust", "payload100.pickle")
    payload = pickle.load(open(path, "rb"))

    sig_d = defaultdict(list)
    for doc_id, q_id, runs in payload:
        print(doc_id, len(runs))

        for sig in ["FBIS", "LA0", "FT9"]:
            if doc_id.startswith(sig):
                sig_d[sig].append(len(runs))

    for sig in sig_d:
        print(sig, average(sig_d[sig]))



if __name__ == '__main__':
    #encode_pred_set(200)
    write_payload_in_plain(200)
