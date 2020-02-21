import os
import pickle

from cpath import output_path
from misc_lib import average


def run():
    filename = "gradient_sero_V.pickle"
    p = os.path.join(output_path, filename)

    data = pickle.load(open(p, "rb"))

    keys = list(data[0].keys())

    for batch in data:
        for key in keys:
            print(key)
            print(average(batch[key]) * 1e8)


run()