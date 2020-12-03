import csv
import sys
from collections import OrderedDict
from typing import Iterable, Tuple

from data_generator.create_feature import create_int_feature
from data_generator.tokenizer_wo_tf import get_tokenizer
from tf_util.record_writer_wrap import RecordWriterWrap
from tlm.data_gen.adhoc_datagen import FirstSegmentAsDoc
from tlm.data_gen.base import get_basic_input_feature


def load_and_format(input_path):
    f = open(input_path, "r", encoding="utf-8", errors="ignore")
    reader = csv.reader(f, delimiter='\t')

    for row in reader:
        query = row[0]
        pos_text = row[1]
        neg_text = row[2]
        yield query, pos_text, 1
        yield query, neg_text, 0


def work(input_file, output_file):
    in_data = load_and_format(input_file)
    writer = RecordWriterWrap(output_file)
    for feature in encode_classification_feature(512, in_data):
        writer.write_feature(feature)
    writer.close()


if __name__ == "__main__":
    work(sys.argv[1], sys.argv[2])


def encode_classification_feature(max_seq_length, data: Iterable[Tuple[str, str, int]]) -> Iterable[OrderedDict]:
    tokenizer = get_tokenizer()
    encoder = FirstSegmentAsDoc(max_seq_length)
    for query, text, label in data:
        q_tokens = tokenizer.tokenize(query)
        text_tokens = tokenizer.tokenize(text)
        input_tokens, segment_ids = encoder.encode(q_tokens, text_tokens)[0]
        feature = get_basic_input_feature(tokenizer, max_seq_length, input_tokens, segment_ids)
        feature['label_ids'] = create_int_feature([label])
        yield feature