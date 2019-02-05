import os
from path import data_path, output_path
from data_generator.tokenizer_b import EncoderUnit
import pickle
import sys



def encode_payload(idx):
    doc_pairs = pickle.load(open("../output/plain_pair_{}.pickle".format(idx), "rb"))
    vocab_filename = "bert_voca.txt"
    voca_path = os.path.join(data_path, vocab_filename)
    max_sequence = 512
    encoder_unit = EncoderUnit(max_sequence, voca_path)

    result = []
    for q1, d1, q2, d2 in doc_pairs:

        for query, text in [(q1,d1), (q2,d2)]:
            entry = encoder_unit.encode_pair(query, text)
            result.append((entry["input_ids"], entry["input_mask"], entry["segment_ids"]))

    filename = os.path.join(output_path, "fad_train_{}.pickle".format(idx))
    pickle.dump(result, open(filename, "wb"))

if __name__ == '__main__':
    encode_payload(int(sys.argv[1]))

