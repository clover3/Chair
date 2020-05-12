import os
import pickle
import sys


def merge_pickle_to_list(format_str, cont_range):
    result = []
    for elem in cont_range:
        pickle_path = format_str.format(elem)
        try:
            result.append(pickle.load(open(pickle_path, "rb")))
        except FileNotFoundError:
            pass

    return result


if __name__ == '__main__':
    dir_path = sys.argv[1]
    format_str = os.path.join(dir_path, "{}")
    st = int(sys.argv[2])
    ed = int(sys.argv[3])
    result = merge_pickle_to_list(format_str, range(st, ed))
    pickle.dump(result, open(sys.argv[4], "wb"))
