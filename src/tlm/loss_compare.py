import pickle
import numpy as np
from path import output_path, data_path
import os
from misc_lib import right
from scipy.stats import ttest_ind

def load_and_compare():
    names = ["dict.pickle", "no_dict.pickle"]
    targets = []
    for name in names:
        p = os.path.join(output_path, name)
        data = pickle.load(open(p, "rb"))
        data = data[0]["masked_lm_example_loss"]
        targets.append((data, name))

    compare(targets[0], targets[1])

def compare(data1 ,data2):
    data1, name1 = data1
    data2, name2 = data2
    print(name1, np.average(data1))
    print(name2, np.average(data2))
    print(ttest_ind(data1, data2))


if __name__ == '__main__':
    load_and_compare()

