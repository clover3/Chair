from collections import OrderedDict

from arg.qck.encode_common import encode_single
from cpath import at_output_dir
from data_generator.tokenizer_wo_tf import get_tokenizer
from dataset_specific.mnli.mnli_reader import MNLIReader, NLIPairData
from dataset_specific.mnli.snli_reader import SNLIReader
from misc_lib import CountWarning
from tf_util.record_writer_wrap import write_records_w_encode_fn
from tlm.data_gen.bert_data_gen import create_int_feature


def get_encode_fn(max_seq_length1, max_seq_length2):
    tokenizer = get_tokenizer()
    count_warning_list = [CountWarning("Prem over length"), CountWarning("Hypo over length")]
    def entry_encode(e: NLIPairData) -> OrderedDict:
        text_list = [e.premise, e.hypothesis]
        max_seq_length_list = [max_seq_length1, max_seq_length2]
        features = OrderedDict()
        for i in range(2):
            max_seq_length = max_seq_length_list[i]
            tokens = tokenizer.tokenize(text_list[i])
            if len(tokens) > max_seq_length-1:
                count_warning_list[i].add_warn()
            input_ids, input_mask, segment_ids = encode_single(tokenizer, tokens, max_seq_length)
            features["input_ids{}".format(i)] = create_int_feature(input_ids)
            features["input_mask{}".format(i)] = create_int_feature(input_mask)
            features["segment_ids{}".format(i)] = create_int_feature(segment_ids)
        features['label_ids'] = create_int_feature([e.get_label_as_int()])
        return features
    return entry_encode


def get_encode_fn2(max_seq_length1, max_seq_length2):
    tokenizer = get_tokenizer()
    cnt_list = [0, 0]
    def entry_encode(e: NLIPairData) -> OrderedDict:
        text_list = [e.premise, e.hypothesis]
        max_seq_length_list = [max_seq_length1, max_seq_length2]
        features = OrderedDict()
        for i in range(2):
            max_seq_length = max_seq_length_list[i]
            tokens = tokenizer.tokenize(text_list[i])
            if len(tokens) > max_seq_length-1:
                cnt_list[i] += 1
        features['label_ids'] = create_int_feature([e.get_label_as_int()])
        return features
    return entry_encode, cnt_list


def gen_mnli(split):
    reader = MNLIReader()
    output_path = at_output_dir("tfrecord", f"nli_p200_h100_{split}")
    entry_encode = get_encode_fn(200, 100)
    write_records_w_encode_fn(output_path, entry_encode, reader.load_split(split), 400 * 1000)


def count_mnli(split):
    reader = MNLIReader()
    output_path = at_output_dir("tfrecord", f"dummy_{split}")
    entry_encode, cnt_list = get_encode_fn2(200, 100)
    write_records_w_encode_fn(output_path, entry_encode, reader.load_split(split), 400 * 1000)
    print(cnt_list)


def gen_snli(split):
    reader = SNLIReader()
    output_path = at_output_dir("tfrecord", f"snli_p200_h100_{split}")
    train_data_size = 549367
    entry_encode = get_encode_fn(200, 100)
    write_records_w_encode_fn(output_path, entry_encode, reader.load_split(split), train_data_size)


def main():
    split = "train"
    count_mnli(split)


if __name__ == "__main__":
    main()