
import sys

from tf_util.enum_features import load_record2


def visualize(filename, n_item):
    for idx, features in enumerate(load_record2(filename)):
        if idx > n_item :
            break
        keys = features.keys()
        for key in keys:
            feature = features[key]
            if feature.int64_list.value:
              values = feature.int64_list.value
            elif feature.float_list.value:
              values = feature.float_list.value
            print("{} : {}".format(key, values[:50]))


if __name__ == "__main__":
    filename = sys.argv[1]
    if len(sys.argv) == 3:
        n_item = int(sys.argv[2])
    else:
        n_item = 1

    visualize(filename, n_item)





