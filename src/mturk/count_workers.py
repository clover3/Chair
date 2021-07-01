import csv
import sys


def get_unique_workers(file_path):
    f = open(file_path, "r", encoding="utf-8")
    data = []
    for row in csv.reader(f):
        data.append(row)
    head = list(data[0])

    column_idx = head.index("WorkerId")
    worker_ids = list([row[column_idx] for row in data[1:]])
    return set(worker_ids)


def main():
    file_path_list_file = sys.argv[1]

    unique_workers = set()
    for line in open(file_path_list_file, 'r'):
        file_path = line.strip()
        try:
            unique_workers.update(get_unique_workers(file_path))
        except ValueError as e:
            print(file_path)
            print(e)
            raise

    print("{} workers in {} files".format(len(unique_workers), "?"))


if __name__ == "__main__":
    main()