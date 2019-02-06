
from data_generator.data_parser.trec import *
from path import data_path
from data_generator.tokenizer_b import EncoderUnit
import sys

def build_tokenzier_cache(job_id):
    vocab_filename = "bert_voca.txt"
    voca_path = os.path.join(data_path, vocab_filename)
    max_sequence = 512

    encoder_unit = EncoderUnit(max_sequence, voca_path)
    full_tokenizer = encoder_unit.encoder.ft

    robust_colleciton = load_robust(robust_path)

    doc_ids = list(robust_colleciton.keys())
    block_size = 10000
    st = job_id * block_size
    ed = (job_id + 1) * block_size

    sub_token_dict = dict()

    for doc_id in doc_ids[st:ed]:
        text = robust_colleciton[doc_id]
        encoder_unit.encoder.encode()
        for token in full_tokenizer.basic_tokenizer.tokenize(text):
            if token not in sub_token_dict:
                sub_tokens = full_tokenizer.wordpiece_tokenizer.tokenize(token)
                sub_token_dict[token] = sub_tokens

    save_to_pickle(sub_token_dict, "sub_tokens_{}.pickle".format(job_id))


if __name__ == '__main__':
    job_id = int(sys.argv[1])
    build_tokenzier_cache(job_id)
