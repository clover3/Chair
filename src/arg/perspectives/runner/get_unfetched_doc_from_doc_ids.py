import sys

from arg.perspectives.clueweb_galago_db import add_doc_list_to_table
from arg.perspectives.runner.get_unfetched_doc_from_q_res import do_join_and_write


def get_doc_list_from_file(file_path):
    doc_ids = []
    f = open(file_path, "r")
    for line in f:
        s = line.strip()
        if s:
            doc_ids.append(s)
    return doc_ids


def work(doc_list_path, save_name):
    doc_list = get_doc_list_from_file(doc_list_path)
    add_doc_list_to_table(doc_list, save_name)
    do_join_and_write(doc_list, save_name)


if __name__ == "__main__":
    work(sys.argv[1], sys.argv[2])
