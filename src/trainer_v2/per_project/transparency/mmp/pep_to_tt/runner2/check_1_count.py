import os
import sys
from collections import Counter


def get_last_line(file_path):
    with open(file_path, 'rb') as f:
        try:  # catch OSError in case of a one line file
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)
        last_line = f.readline().decode()

        return last_line


def enum_files(directory):
    counter = Counter()
    for i in range(0, 100000):
        file_path = f"{directory}/{i}.txt"
        try:
            line = get_last_line(file_path)
            if not line.strip():
                continue

            term, score_s = line.split("\t")
            is_one = float(score_s) > 0.9999
            if is_one:
                counter["one"] += 1
            else:
                counter["not_one"] += 1
        except FileNotFoundError:
            pass
        except ValueError as e:
            print(file_path)
            print(e)

    print(counter)

# Use the function
directory = sys.argv[1]
enum_files(directory)

