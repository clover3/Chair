import os
from data_generator.tokenizer_b import EncoderUnit
from path import data_path
import pickle
import sys

def encode(input_path, output_path):
    payload = pickle.load(open(input_path, "rb"))
    vocab_filename = "bert_voca.txt"
    voca_path = os.path.join(data_path, vocab_filename)
    max_sequence = 512
    encoder_unit = EncoderUnit(max_sequence, voca_path)
    result = []
    print("Encode...")
    for q_id, doc_id, doc_query_list in payload:
        runs = []
        for query, sent in doc_query_list:
            entry = encoder_unit.encode_pair(query, sent)
            runs.append(entry)
        result.append((doc_id, q_id, runs))
    pickle.dump(result, open(output_path, "wb"))
    print("Done...")


if __name__ == '__main__':
    data_id = int(sys.argv[1])
    print(data_id)
    input_path = "/mnt/nfs/work3/youngwookim/code/Chair/data/robust/payload_512_2k_plain_parts/payload512_part_{}".format(data_id)
    output_path = "/mnt/nfs/work3/youngwookim/code/Chair/data/robust/payload_512_2k_encoded_parts/enc_payload512_part_{}".format(data_id)
    encode(input_path, output_path)

