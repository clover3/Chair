import os
import sys


def enum_files(directory):
    missing = []
    for i in range(0, 100000):
        if i // 10 in missing:
            continue
        file_path = f"{directory}/{i}.txt"
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            # print("Not done at {}".format(i))
            missing.append(i // 10)
    print(" ".join(map(str, missing)))

# Use the function
directory = sys.argv[1]
enum_files(directory)

