import csv
import sys

from tf_util.record_writer_wrap import RecordWriterWrap
from tlm.data_gen.classification_common import encode_classification_feature


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

