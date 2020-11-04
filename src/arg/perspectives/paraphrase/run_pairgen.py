import os
from collections import OrderedDict

from arg.perspectives.evaluate import perspective_getter
from arg.perspectives.load import splits
from arg.perspectives.paraphrase.pairgen import generate_pair_insts, Instance
from cpath import output_path
from data_generator.tokenizer_wo_tf import get_tokenizer
from misc_lib import exist_or_mkdir
from tf_util.record_writer_wrap import write_records_w_encode_fn
from tlm.data_gen.base import get_basic_input_feature
from tlm.data_gen.bert_data_gen import create_int_feature


def encode(tokenizer, get_tokens, max_seq_length, inst: Instance) -> OrderedDict:
    tokens1 = get_tokens(inst.pid1)
    tokens2 = get_tokens(inst.pid2)
    tokens = ["[CLS]"] + tokens1 + ["[SEP]"] + tokens2 + ["[SEP]"]
    segment_ids = [0] * (len(tokens1) + 2) \
                  + [1] * (len(tokens2) + 1)
    tokens = tokens[:max_seq_length]
    segment_ids = segment_ids[:max_seq_length]
    features = get_basic_input_feature(tokenizer, max_seq_length, tokens, segment_ids)
    features['label_ids'] = create_int_feature([inst.label])
    return features


def main():
    dir_path = os.path.join(output_path, "perspective_paraphrase")
    seq_length = 100
    tokenizer = get_tokenizer()
    tokens_d = {}

    def get_tokens(pid):
        if pid not in tokens_d:
            text = perspective_getter(pid)
            tokens_d[pid] = tokenizer.tokenize(text)

        return tokens_d[pid]

    def encode_fn(inst):
        return encode(tokenizer, get_tokens, seq_length, inst)

    exist_or_mkdir(dir_path)
    for split in splits:
        insts = generate_pair_insts(split)
        save_path = os.path.join(dir_path, split)
        write_records_w_encode_fn(save_path, encode_fn, insts)


if __name__ == "__main__":
    main()