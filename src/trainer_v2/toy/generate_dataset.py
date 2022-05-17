import random
from collections import OrderedDict

from cpath import at_output_dir
from data_generator.create_feature import create_int_feature
from tf_util.record_writer_wrap import RecordWriterWrap


def enc_to_feature(i) -> OrderedDict:
    n_dim = 10
    x = [random.randint(1, 8) for _ in range(n_dim)]
    y = sum(x)
    feature = OrderedDict()
    feature["x"] = create_int_feature(x)
    feature["y"] = create_int_feature([y])
    return feature


def main():
    n_data = 1000
    outpath = at_output_dir("tfrecord", "toy.tfrecord")
    writer = RecordWriterWrap(outpath)
    for i in range(n_data):
        writer.write_feature(enc_to_feature(i))
    writer.close()


if __name__ == "__main__":
    main()