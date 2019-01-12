from data_generator.adhoc.data_sampler import DataSampler, DataWriter
from data_generator.data_parser.robust import *

from concurrent.futures import ProcessPoolExecutor, wait
from data_generator.data_parser import trec
from data_generator.tokenizer_b import EncoderUnit
from path import data_path

import pickle

def encode_pred_set(top_k):
    vocab_filename = "bert_voca.txt"
    max_sequence = 200
    voca_path = os.path.join(data_path, vocab_filename)
    encoder_unit = EncoderUnit(max_sequence, voca_path)
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


if __name__ == '__main__':
    encode_pred_set(100)

