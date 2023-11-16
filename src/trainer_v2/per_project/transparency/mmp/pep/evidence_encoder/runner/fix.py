import pickle
import sys

import numpy as np



def fix(item):
    (a, b), b_ = item
    assert type(b) == list
    assert type(b[0]) == np.ndarray
    return a, b




def main():
    file_path = sys.argv[1]
    obj = pickle.load(open(file_path, "rb"))
    new_obj = list(map(fix, obj))
    pickle.dump(new_obj, open(file_path, "wb"))


if __name__ == "__main__":
    main()