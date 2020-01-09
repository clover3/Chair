import csv
import sys

from scipy.stats import ttest_ind


def load_file(file_path):
    f = open(file_path, "r")
    reader = csv.reader(f, delimiter=',')

    for idx, row in enumerate(reader):
        if idx == 0: continue  # skip header
        # Works for both splits even though dev has some extra human labels.
        yield row

def compare(file1, file2):
    all_steps = []

    value_d_1 = {}
    value_d_2 = {}
    for _, step, value in load_file(file1):
        step = int(step)
        all_steps.append(step)
        value_d_1[step] = float(value)


    for _, step, value in load_file(file2):
        step = int(step)
        all_steps.append(step)
        value_d_2[step] = float(value)

    print(len(all_steps))


    all_steps.sort()

    parallel_data = []
    for step in all_steps:
        if step in value_d_1 and step in value_d_2:
            parallel_data.append((value_d_1[step], value_d_2[step]))

    print(parallel_data[:10])
    print(len(parallel_data))

    data1, data2 = zip(*parallel_data)
    return ttest_ind(data1, data2)

if __name__ == "__main__":
    r = compare(sys.argv[1], sys.argv[2])
    print(r)
