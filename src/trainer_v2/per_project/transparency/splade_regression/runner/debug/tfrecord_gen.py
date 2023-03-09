from collections import OrderedDict

from data_generator.create_feature import create_float_feature
from tf_util.record_writer_wrap import write_records_w_encode_fn


def encode_fn(_) -> OrderedDict:
    features = OrderedDict()
    features["x"] = create_float_feature([0.1] * 10)
    features["y"] = create_float_feature([0.1])
    return features


def main():
    save_path = "debug_random_float"
    write_records_w_encode_fn(save_path, encode_fn, range(1000))


if __name__ == "__main__":
    main()