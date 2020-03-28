import sys

from datastore.interface import get_existing_keys
from datastore.table_names import RawCluewebDoc


def work(file_path):
    doc_ids = list([line.strip() for line in open(file_path, "r")])
    found_ids = set(get_existing_keys(RawCluewebDoc, doc_ids))

    for doc_id in doc_ids:
        if doc_id not in found_ids:
            print(doc_id)


if __name__ == "__main__":
    work(sys.argv[1])

