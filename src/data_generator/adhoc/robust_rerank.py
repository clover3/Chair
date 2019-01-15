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

    payload = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        future_list = []
        for q_id in ranked_lists:
            ranked = ranked_lists[q_id]
            ranked.sort(key=lambda x:x[1])
            assert ranked[0][1] == 1

            doc_ids = []
            for doc_id, rank, score, in ranked[:top_k]:
                doc_ids.append(doc_id)

                raw_query = queries[q_id]
                content = collection[doc_id]
                runs_future = executor.submit(encoder_unit.encode_long_text, raw_query, content)
                future_list.append((doc_id, q_id, runs_future))
        for doc_id, q_id, runs_future in future_list:
            payload.append((doc_id, q_id, runs_future.result()))

    pickle.dump(payload, open("payload{}.pickle".format(top_k), "wb"))


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
    encode_pred_set(200)

