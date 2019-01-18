import os
from data_generator.tokenizer_b import EncoderUnit
from path import data_path
import pickle


def encode(input_path, output_path):
    payload = pickle.load(open(input_path, "rb"))
    vocab_filename = "bert_voca.txt"
    voca_path = os.path.join(data_path, vocab_filename)
    max_sequence = 200
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
    data_id = 0
    input_path = "REPLACE/payload200_part_{}".format(data_id)
    output_path = "REPLACE/enc_payload200_part_{}".format(data_id)
    encode(input_path, output_path)