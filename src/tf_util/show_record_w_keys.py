import argparse
import sys
import tensorflow as tf
parser = argparse.ArgumentParser(description='File should be stored in ')
parser.add_argument("--tfrecord_path", default="")
parser.add_argument("--int_features", default="")
parser.add_argument("--float_features", default="")

def file_show():
    args = parser.parse_args(sys.argv[1:])

    def parse_list(s):
        return [t.strip() for t in s.split(",")]

    int_features = parse_list(args.int_features)
    float_features = parse_list(args.float_features)
    cnt = 0
    n_display = 5
    for record in tf.compat.v1.python_io.tf_record_iterator(args.tfrecord_path):
        example = tf.train.Example()
        example.ParseFromString(record)
        feature = example.features.feature
        keys = feature.keys()

        print("---- record -----")
        for key in keys:
            if key in float_features:
                v = feature[key].float_list.value
            elif key in int_features:
                v = feature[key].int64_list.value
            else:
                v = feature[key].int64_list.value

            print(key)
            print(v)

        cnt += 1
        if cnt >= n_display:  ##
            break


if __name__ == "__main__":
    file_show()
